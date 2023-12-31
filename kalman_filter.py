import numpy as np
import math as math


class KalmanFilter(object):
    # x is the state vector, z measurement and u control vector
    # Model being used is in form :
    # x = Fx + Gu + w, z = Hx + v
    # w and v have covariance matrices Q and R
    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = np.zeros((dim_x, ))        # state
        self.P = np.eye(dim_x)              # uncertainty covariance
        self.Q = np.eye(dim_x)              # process uncertainty
        self.B = None                       # control transition matrix
        self.F = np.eye(dim_x)              # state transition matrix
        self.H = np.zeros((dim_z, dim_x))   # measurement function
        self.R = np.eye(dim_z)              # measurement uncertainty
        self.z = np.array((dim_z, ))

        self.K = np.zeros((dim_x, dim_z))   # Kalman gain
        self.v = np.zeros((dim_z, ))        # innovation
        self.S = np.zeros((dim_z, dim_z))   # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        self._I = np.eye(dim_x)             # identity matrix

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self.inv = np.linalg.inv

    def predict(self, u=None, B=None, F=None, Q=None):
        # Prediction step of KF algorithm
        # Prediction is calculated as expected value of the model, conditioned by the measurements
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self.dim_x)*Q

        # x_hat = Fx + Bu, it is assumed that noise is 0 mean

        if B is not None and u is not None:
            self.x = F @ self.x + B @ self.x
        else:
            self.x = F @ self.x

        # Need to update the uncertainty, P = FPF' + Q

        self.P = F @ self.P @ F.T + Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, R = None, H = None):
        # update stage of the filtering process
        # final estimate is calculated as : x_estimate = x_estimate_old K*innovation where the innovation is given by:
        # innovation = (z_measurement - Hx_prediction), K is kalman gain, it is derived by minimizing the mse
        # K = PH'*S^-1, where the S is the innovation uncertainty, S = HPH'+R

        if z is None:
            self.z = np.array([None]*self.dim_z)
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.v = np.zeros((self.dim_z, ))
            return

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z)*R
        if H is None:
            H = self.H

        # innovation calculation:
        self.v = z - H @ self.x
        PHT = self.P @ H.T

        # now the innovation uncertainty: S = HPH' + R
        self.S = H @ PHT + R
        self.SI = self.inv(self.S)

        # Now to calculate the Kalman gain
        self.K = PHT @ self.SI

        # final prediction can be made as x = x + K*innovation
        self.x = self.x + self.K @ self.v

        # P = (I-KH)P(I-KH)' + KRK' a more numerically stable version of P = (I-KH)P
        I_KH = self._I - self.K @ H
        self.P = I_KH @ self.P @ I_KH.T + self.K @ R @ self.K.T
        # save measurement and posterior state
        self.z = np.copy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()


class KalmanInformationFilter(object):

    # actually inverse of the Kalman filter, allowing you to easily denote having
    # no information at initialization.

    def __init__(self, dim_x, dim_z, dim_u=0):

        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = np.zeros((dim_x, ))  # state
        self.x_info = np.zeros((dim_x, ))  # state in information space
        self.P_inv = np.eye(dim_x)   # uncertainty covariance
        self.Q = np.eye(dim_x)       # process uncertainty
        self.B = 0.               # control transition matrix
        self.F = 0.              # state transition matrix
        self._F_inv = 0.          # state transition matrix
        self.H = np.zeros((dim_z, dim_x))  # Measurement function
        self.R_inv = np.eye(dim_z)   # state uncertainty
        self.z = np.array((dim_z, ))

        self.K = 0.  # kalman gain
        self.v = np.zeros((dim_z, ))  # innovation
        self.S = 0.  # system uncertainty in measurement space

        # identity matrix.
        self._I = np.eye(dim_x)
        self.inv = np.linalg.inv

        # save priors and posteriors
        self.x_prior = np.copy(self.x)
        self.P_inv_prior = np.copy(self.P_inv)
        self.x_post = np.copy(self.x)
        self.P_inv_post = np.copy(self.P_inv)


    def update(self, z, R_inv=None, H = None, multiple_sensors = False):
        # update stage of the filtering process
        # estimation is preformed in information space as: y = y + i, where i is the information contribution
        # if there are multiple sensors information is additive. i = H'R_inv z
        # Total information content can be updated as Pinv = Pinv + I, I = H'R_invH

        if z is None:
            self.z = np.array([None] * self.dim_z)
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.v = np.zeros((self.dim_z, ))
            return

        if R_inv is None:
            R_inv = self.R_inv
        elif np.isscalar(R_inv):
            R_inv = np.eye(self.dim_z) * R_inv

        if H is None:
            H = self.H

        H_T = H.T
        if multiple_sensors:
            # It is assumed that data has been processed externally for None values, reason is that it's much easier
            # to do it apriori.
            z = z.reshape((*z.shape, 1))  # reshaping into a 3D array of form N x dim_z x 1 for broadcasting purposes
            # R_inv will have the shape N x dim_z x dim_z in this case
        else:

            R_inv = R_inv.reshape((1, *R_inv.shape))  # reshaping to make the following code work for both cases, now E_inv has
            # shape (1, dim_z, dim_z)
            z = z.reshape((1, self.dim_z, 1))

        P_inv = self.P_inv

        ik = H_T @ (R_inv @ z).sum(axis=0)  # sensor information contribution
        ik = ik.squeeze()
        Ik = (H_T @ R_inv @ H).sum(axis=0)  # sensor uncertainty contribution

        self.x_info += ik
        self.P_inv += Ik
        self.x = np.linalg.solve(P_inv, self.x_info)

        # save measurement and posterior state
        self.z = np.copy(z)
        self.x_post = self.x.copy()
        self.P_inv_post = self.P_inv.copy()

    def predict(self, u=None, B=None, F=None, Q=None):
        # Prediction step of KF algorithm
        # Prediction is calculated as expected value of the model, conditioned by the measurements
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self.dim_x) * Q

        # x_hat = Fx + Bu, it is assumed that noise is 0 mean

        if B is not None and u is not None:
            self.x = np.dot(F, self.x) + np.dot(B, u)
        else:
            self.x = np.dot(F, self.x)

        # Need to update the uncertainty, P_inv = (FPF' + Q)^-1

        self.P_inv = self.inv(F @ self.inv(self.P_inv) @ F.T + Q)

        # In information space x_info = Pinv*x
        self.x_info = np.dot(self.P_inv, self.x)

        # save prior
        self.x_prior = self.x.copy()
        self.P_inv_prior = self.P_inv.copy()
