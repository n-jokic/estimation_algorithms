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

        self.P = np.dot(dxf(self.x), self.P).dot(dxf(self.x).T) + Q
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
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.v = np.zeros((self.dim_z, 1))
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
        PHT = np.dot(self.P, dxh(self.x).T)

        # now the innovation uncertainty: S = dxh P dhx' + R dhx is jacobian of h evaluated at predicted value of x
        self.S = np.dot(dxh(self.x), PHT) + R
        self.SI = self.inv(self.S)

        # Now to calculate the Kalman gain
        self.K = np.dot(PHT, self.SI)

        # final prediction can be made as x = x + K*innovation
        self.x = self.x + np.dot(self.K, self.v)

        # P = P - KSK'

        self.P = self.P - np.dot(self.K, self.S).dot(self.K.T)
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

        self.x = np.zeros((dim_x, 1))  # state
        self.x_info = np.zeros((dim_x, 1))  # state in information space
        self.P_inv = np.eye(dim_x)  # estimation information
        self.f = None  # non-linear process model
        self.h = None  # non-linear measurement model
        self.dxf = None  # process model Jacobian
        self.dxh = None  # measurement model Jacobian
        self.R_inv = np.eye(dim_z)  # measurement noise
        self.Q = np.eye(dim_x)  # process noise

        self.v = np.zeros((dim_z, 1))  # innovation
        self.z = np.zeros((dim_z, 1))  # measurement

        self.K = np.zeros((dim_x, dim_z))  # Kalman gain matrix
        self.S = np.zeros((dim_z, dim_z))  # innovation covariance in information space

        self._I = np.eye(dim_x)  # identity matrix
        self.inv = np.linalg.inv

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_inv_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_inv_post = self.P.copy()

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

        self.P_inv = self.inv(np.dot(dxf(self.x), self.inv(self.P_inv)).dot(dxf(self.x).T) + Q)
        self.x_info = x_info
        self.x = np.linalg.solve(self.P_inv, self.x_info)

        # save prior
        self.x_prior = self.x.copy()
        self.P_inv_prior = self.P.copy()

    def update(self, z, R_inv=None, h=None, dxh=None, multiple_sensors=False):
        # update stage of the filtering process
        # final estimate is calculated as : x_estimate = x_estimate_old K*innovation where the innovation is given by:
        # innovation = (z_measurement - Hx_prediction), K is kalman gain, it is derived by minimizing the mse
        # K = PH'*S^-1, where the S is the innovation uncertainty, S = HPH'+R

        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_inv_post = self.P_inv.copy()
            self.v = np.zeros((self.dim_z, 1))
            return

        if R_inv is None:
            R_inv = self.R_inv
        elif np.isscalar(R_inv):
            R_inv = np.eye(self.dim_z) * R_inv

        if h is None:
            h = self.h
        if dxh is None:
            dxh = self.dxh

        number_of_sensors = z.shape[1]
        jacobian_h = dxh(self.x)
        z_hat = h(self.x)
        bias = jacobian_h.dot(self.x)

        ik = 0  # sensor information contribution
        Ik = 0  # sensor uncertainty contribution

        if multiple_sensors:
            for i in range(number_of_sensors):
                R_inv_cur = R_inv[i]
                c = z[:, i].reshape((self.dim_z, 1))
                # innovation calculation:
                self.v = c - z_hat
                ik += np.dot(jacobian_h.T, R_inv_cur).dot(self.v + bias)
                Ik += np.dot(jacobian_h.T, R_inv_cur).dot(jacobian_h)
        else:
            self.v = z - z_hat
            ik += np.dot(jacobian_h.T, R_inv).dot(self.v + bias)
            Ik += np.dot(jacobian_h.T, R_inv).dot(jacobian_h)

        # final prediction can be made as x = x + sum(ik)
        self.x_info += ik

        # P = P + sum(Ik)

        self.P_inv += Ik
        # save measurement and posterior state
        self.z = np.copy(z)
        self.x = np.linalg.solve(self.P_inv, self.x_info)

        self.x_post = self.x.copy()
        self.P_inv_post = self.P_inv.copy()