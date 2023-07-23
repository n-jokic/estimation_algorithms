import numpy as np


class ExtendedKalmanFilter(object):
    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_u = dim_u
        self.dim_z = dim_z
        self.dim_x = dim_x

        self.x = np.zeros((dim_x, ))  # state
        self.P = np.eye(dim_x)  # estimation uncertainty
        self.f = None  # non-linear process model
        self.h = None  # non-linear measurement model
        self.dxf = None  # process model Jacobian
        self.dxh = None  # measurement model Jacobian
        self.R = np.eye(dim_z)  # measurement noise
        self.Q = np.eye(dim_x)  # process noise

        self.v = np.zeros((dim_z, ))  # innovation
        self.z = np.zeros((dim_z, ))  # measurement

        self.K = np.zeros((dim_x, dim_z))  # Kalman gain matrix
        self.S = np.zeros((dim_z, dim_z))  # innovation covariance
        self.SI = np.zeros((dim_z, dim_z))  # innovation covariance inversion

        self._I = np.eye(dim_x)  # identity matrix
        self.inv = np.linalg.inv

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict(self, u=None, f=None, dxf=None, Q=None):
        # Prediction step of KF algorithm
        # Prediction is calculated as expected value of the model, conditioned by the measurements
        if f is None:
            f = self.f
        if dxf is None:
            dxf = self.dxf
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self.dim_x) * Q

        # x_hat = Fx + Bu, it is assumed that noise is 0 mean

        if u is not None:
            x = f(self.x, u)
        else:
            x = f(self.x)

        # Need to update the uncertainty, P = dxf P dxf' + Q, dxf is the jacobian of f,
        # jacobian is evaluated at previous value of x

        self.P = dxf(self.x) @ self.P @ dxf(self.x).T + Q
        self.x = x

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, R=None, h=None, dxh=None):
        # update stage of the filtering process
        # final estimate is calculated as : x_estimate = x_estimate_old K*innovation where the innovation is given by:
        # innovation = (z_measurement - Hx_prediction), K is kalman gain, it is derived by minimizing the mse
        # K = PH'*S^-1, where the S is the innovation uncertainty, S = HPH'+R

        if z is None:
            self.z = np.array([None] * self.dim_z)
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.v = np.zeros((self.dim_z, ))
            return

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        if h is None:
            h = self.h
        if dxh is None:
            dxh = self.dxh

        # innovation calculation:
        self.v = z - h(self.x)
        PHT = self.P @ dxh(self.x).T

        # now the innovation uncertainty: S = dxh P dhx' + R dhx is jacobian of h evaluated at predicted value of x
        self.S = dxh(self.x) @ PHT + R
        self.SI = self.inv(self.S)

        # Now to calculate the Kalman gain
        self.K = PHT @ self.SI

        # final prediction can be made as x = x + K*innovation
        self.x = self.x + self.K @ self.v

        # P = P - KSK'

        self.P = self.P - self.K @ self.S @ self.K.T
        # save measurement and posterior state
        self.z = np.copy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

class ExtendedKalmanInformationFilter(object):
    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_u = dim_u
        self.dim_z = dim_z
        self.dim_x = dim_x

        self.x = np.zeros((dim_x, ))  # state
        self.x_info = np.zeros((dim_x, ))  # state in information space
        self.P_inv = np.eye(dim_x)  # estimation information
        self.f = None  # non-linear process model
        self.h = None  # non-linear measurement model
        self.dxf = None  # process model Jacobian
        self.dxh = None  # measurement model Jacobian
        self.R_inv = np.eye(dim_z)  # measurement noise
        self.Q = np.eye(dim_x)  # process noise

        self.v = np.zeros((dim_z, ))  # innovation
        self.z = np.zeros((dim_z, ))  # measurement

        self.K = np.zeros((dim_x, dim_z))  # Kalman gain matrix
        self.S = np.zeros((dim_z, dim_z))  # innovation covariance in information space

        self._I = np.eye(dim_x)  # identity matrix
        self.inv = np.linalg.inv

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_inv_prior = self.P_inv.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_inv_post = self.P_inv.copy()

    def predict(self, u=None, f=None, dxf=None, Q=None):
        # Prediction step of KF algorithm
        # Prediction is calculated as expected value of the model, conditioned by the measurements
        if f is None:
            f = self.f
        if dxf is None:
            dxf = self.dxf
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self.dim_x) * Q

        # x_hat = Fx + Bu, it is assumed that noise is 0 mean

        if u is not None:
            x_info = f(self.x, u)
        else:
            x_info = f(self.x)

        # Need to update the uncertainty, P = dxf P dxf' + Q, dxf is the jacobian of f,
        # jacobian is evaluated at previous value of x

        self.P_inv = self.inv(dxf(self.x) @ self.inv(self.P_inv) @ dxf(self.x).T + Q)
        self.x_info = x_info
        self.x = np.linalg.solve(self.P_inv, self.x_info)

        # save prior
        self.x_prior = self.x.copy()
        self.P_inv_prior = self.P_inv.copy()

    def update(self, z, R_inv=None, h=None, dxh=None, multiple_sensors=False):
        # update stage of the filtering process
        # final estimate is calculated as : x_estimate = x_estimate_old K*innovation where the innovation is given by:
        # innovation = (z_measurement - Hx_prediction), K is kalman gain, it is derived by minimizing the mse
        # K = PH'*S^-1, where the S is the innovation uncertainty, S = HPH'+R

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_inv_post = self.P_inv.copy()
            self.v = np.zeros((self.dim_z, ))
            return

        if R_inv is None:
            R_inv = self.R_inv
        elif np.isscalar(R_inv):
            R_inv = np.eye(self.dim_z) * R_inv

        if h is None:
            h = self.h
        if dxh is None:
            dxh = self.dxh

        if multiple_sensors:
            number_of_sensors = z.shape[0]   # It is assumed that measurements are stacked in rows
            self.v = np.zeros((number_of_sensors, self.dim_z))
            jacobian_h = np.zeros((number_of_sensors, self.dim_z, self.dim_x))
            bias = np.zeros((number_of_sensors, self.dim_z))
            for idx in range(number_of_sensors):
                h_cur = h[idx]
                dxh_cur = dxh[idx]
                self.v[idx, :] = z[idx, :] - h_cur(self.x)
                jacobian_h[idx, :, :] = dxh_cur(self.x)
                bias[idx, :] = jacobian_h[idx, :, :] @ self.x
            self.v = self.v.reshape((*self.v.shape, 1))
            bias = bias.reshape((*bias.shape, 1))
        else:
            self.v = np.zeros((1, self.dim_z))
            jacobian_h = np.zeros((1, self.dim_x, self.dim_x))
            bias = np.zeros((1, self.dim_z))
            self.v[0, :] = z - h(self.x)
            jacobian_h[0, :, :] = dxh(self.x)
            bias[0, :] = jacobian_h[0, :, :] @ self.x
            self.v = self.v.reshape((*self.v.shape, 1))
            bias = bias.reshape((*bias.shape, 1))
            R_inv = R_inv.reshape((1, *R_inv.shape))

        jacobian_h_T = jacobian_h.transpose(0, 2, 1)  # Transposing only the matrices
        ik = (jacobian_h_T @ R_inv @ (self.v + bias)).sum(axis=0)  # sensor information contribution
        ik = ik.squeeze()
        Ik = (jacobian_h_T @ R_inv @jacobian_h).sum(axis=0)  # sensor uncertainty contribution

        # final prediction can be made as x = x + sum(ik)
        self.x_info += ik

        # P = P + sum(Ik)
        self.P_inv += Ik

        # save measurement and posterior state
        self.z = np.copy(z)
        self.x = np.linalg.solve(self.P_inv, self.x_info)

        self.x_post = self.x.copy()
        self.P_inv_post = self.P_inv.copy()