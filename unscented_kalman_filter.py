import numpy as np


class UnscentedKalmanFilter(object):

    def __init__(self, dim_x=1, dim_z=1, dim_u=0):

        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # Parameters for sigma point generation :
        self._alpha = 0.001
        self._beta = 2
        self._k = 3 - dim_x

        self.x = np.zeros((dim_x, 1))  # state
        self.P = np.eye(dim_x)  # estimation uncertainty covariance
        self.P_chol = np.eye(dim_x)  # Cholesky decomposition of P
        self.Q = np.eye(dim_x)  # process uncertainty
        self.f = None  # non-linear process function

        self.z = np.zeros((dim_z, 1))  # measurement
        self.h = None  # non-linear measurement function
        self.R = np.eye(dim_z)  # measurement uncertainty

        self.K = np.zeros((dim_x, dim_z))  # Kalman gain matrix
        self.v = np.zeros((dim_z, 1))  # innovation vector
        self.S = np.zeros((dim_z, dim_z))  # innovation covariance
        self.SI = np.zeros((dim_z, dim_z))  # innovation covariance inverse
        self.C = np.zeros((dim_x, dim_z))  # measurement and state cross-covariance

        self._I = np.eye(dim_x)  # identity matrix

        # values of x after prediction step
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        self.Wp = np.ones((1, 2 * dim_x + 1))  # weights for sigma points for state estimation
        self.Wc = np.ones((1, 2 * dim_x + 1))  # weights for sigma points for covariance estimation
        self._update_weights()

        # values of x after update step
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self.inv = np.linalg.inv
        self.cholesky = np.linalg.cholesky
        self.eps = np.eye(dim_x) * 0.0001
        self.max_iter = 10

        self._recalculate_weights = False

    @property  # getter property for alpha
    def alpha(self):
        return self._alpha

    @property  # getter property for alpha
    def beta(self):
        return self._beta

    @property  # getter property for alpha
    def k(self):
        return self._k

    @alpha.setter
    def alpha(self, value):
        if not np.isscalar(value):
            raise ValueError('alpha must be a scalar')
        if value < 1e-4:
            raise ValueError('alpha must be greater than 1e-4')
        if value > 1:
            raise ValueError('alpha must be less than 1')
        self._alpha = value
        self._update_weights()

    @k.setter
    def k(self, value):
        if not np.isscalar(value):
            raise ValueError('k must be a scalar')
        self._k = value
        self._update_weights()

    @beta.setter
    def beta(self, value):
        if not np.isscalar(value):
            raise ValueError('beta must be a scalar')
        self._beta = value
        self._update_weights()

    def _update_weights(self):
        alpha = self.alpha
        beta = self.beta
        k = self.k
        n = self.dim_x

        lamb = (n + k) * alpha ** 2 - n

        Wp = np.ones((1, 2 * n + 1)) / 2 / (n + lamb)
        Wc = Wp.copy()
        Wp[0, 0] *= 2 * lamb
        Wc[0, 0] = Wp[0, 0] + 1 - alpha ** 2 + beta

        self.Wp = Wp
        self.Wc = Wc

    def _safe_cholesky(self):  # This function ensures that P_chol > 0
        eps = self.eps
        max_iter = self.max_iter

        for idx in range(max_iter):
            P_chol = self.cholesky(self.P + idx * eps)
            eig_values = np.linalg.eig(P_chol)

            # if any(any(np.real(eig) < 0) for eig in eig_values):
            #    continue  # we need to add 1 more eps, P_chol must have positive eig values

            return P_chol

        raise ValueError('P_cholesky is ill-conditioned, increase max_iter')

    def _generate_sigma_points(self):

        alpha = self.alpha
        k = self.k

        n = self.dim_x
        lamb = (alpha ** 2) * (n + k) - n
        a = np.sqrt(n + lamb)
        X = np.zeros((n, 2 * n + 1))

        P_chol = a * self._safe_cholesky()
        X[:, 0] = self.x[:, 0]
        X[:, 1:n + 1] = (self.x + P_chol)
        X[:, n + 1:2 * n + 1] = (self.x - P_chol)

        Wc = self.Wc
        Wp = self.Wp

        return X, Wc, Wp

    def prediction(self, u=None, f=None, Q=None):
        if f is None:
            f = self.f
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self.dim_x) * Q

        # First we need to generate 2n + 1 sigma points, n is number of states in the model
        # Wc are weights used for covariance estimation
        # Wp are weights used for mean estimation
        X, Wc, Wp = self._generate_sigma_points()
        # Now we need to propagate the points through the process model
        if u is None:
            X = f(X)
        else:
            if np.isscalar(u):
                u = np.eye(self.dim_u) * u
            X = f(X, u)

        # Now we can use the propagated sigma points for mean and covariance estimation:
        # x = sum(Wp_iX_i)                                 equation for mean calculation
        # P = sum(Wc_i(X_i - x)(X_i - x)')           equation for covariance calculation

        x = (X * Wp).sum(axis=1, keepdims=True)  # Python magic: Wp has the shape (1, 2n+1) and X has shape (n, 2n+1)
        # as a first step Wp*X will result in matrix with shape (n, 2n+1), each row from X will get multiplied
        # element wise with values from row vector Wp, next we want to sum all the columns, that can be achieved
        # by using the np.sum(Wp*X, axis=1), if axis=0 then rows are summed, only issue is that np.sum will return
        # a vector with dimensions (n, ) so we have to use keepdims=True to preserve the dimension (n, 1)

        # More info about broadcasting can be found here: https://numpy.org/doc/stable/user/basics.broadcasting.html

        self.S = ((X - x) * Wc).dot((X - x).T) + Q  # Similarly like before (X-x)*Wp multiples each row of (X-x) with
        # values from row vector Wc.

        self.x = x

        self.x_prior = x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, u=None, h=None, R=None):

        if z is None:  # No measurement is available
            self.x_post = self.x
            self.P_post = self.P
            return

        if h is None:
            h = self.h
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        # First we need to generate 2n + 1 sigma points, n is number of states in the model
        # Wc are weights used for covariance estimation
        # Wp are weights used for mean estimation
        X, Wc, Wp = self._generate_sigma_points()
        x = self.x
        P = self.P

        # Now we need to propagate the points through the measurement model
        if u is None:
            Z = h(X)
        else:
            Z = h(X, u)

        # Now we can use the propagated sigma points for mean and covariance estimation:
        # z_hat = sum(Wp_iX_i)                             equation for mean calculation
        # S = sum(Wc_i(Z_i - z_hat)(Z_i - z_hat)')           equation for covariance calculation
        # C = sum(Wc_i(X_i - x)(Z_i - z_hat)')

        z_hat = np.sum(Z * Wp, axis=1,
                       keepdims=True)  # Python magic: Wp has the shape (1, 2n+1) and X has shape (n, 2n+1)
        # as a first step Wp*X will result in matrix with shape (n, 2n+1), each row from X will get multiplied
        # element wise with values from row vector Wp, next we want to sum all the columns, that can be achieved
        # by using the np.sum(Wp*X, axis=1), if axis=0 then rows are summed, only issue is that np.sum will return
        # a vector with dimensions (n, ) so we have to use keepdims=True to preserve the dimension (n, 1)

        # More info about broadcasting can be found here: https://numpy.org/doc/stable/user/basics.broadcasting.html

        S = ((Z - z_hat) * Wc).dot((Z - z_hat).T) + R  # Similarly like before (X-x)*Wp multiples each row of (X-x) with
        S = S / 2 + S.T / 2
        # values from row vector Wc.
        C = ((X - x) * Wc).dot((Z - z_hat).T)
        SI = self.inv(S)

        # Now we can calculate the Kalman gain:
        # K = C*SI and use it for final update:
        # x = x + K(z - z_hat), v = z - z_hat -> innovation
        # P = P - KSK' - covariance update, to ensure symmetry and numerical stability we will preform
        # correction : P = P/2 + P'/2

        v = z - z_hat
        K = C * SI
        x = x + K * v
        P = P - K * S * K.T
        P = P / 2 + P.T / 2 + self.eps

        self.x = x
        self.v = v
        self.z = z
        self.K = K
        self.S = S
        self.SI = SI
        self.C = C
        self.P = P

        self.x_post = x
        self.P_post = P


class UnscentedKalmanInformationFilter(object):

    def __init__(self, dim_x=1, dim_z=1, dim_u=0):

        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # Parameters for sigma point generation :
        self._alpha = 0.001
        self._beta = 2
        self._k = 3 - dim_x

        self.x = np.zeros((dim_x, 1))  # state
        self.P = np.eye(dim_x)  # estimation uncertainty covariance
        self.P_chol = np.eye(dim_x)  # Cholesky decomposition of P
        self.Q = np.eye(dim_x)  # process uncertainty
        self.f = None  # non-linear process function

        self.z = np.zeros((dim_z, 1))  # measurement
        self.h = None  # non-linear measurement function
        self.R = np.eye(dim_z)  # measurement uncertainty

        self.K = np.zeros((dim_x, dim_z))  # Kalman gain matrix
        self.v = np.zeros((dim_z, 1))  # innovation vector
        self.S = np.zeros((dim_z, dim_z))  # innovation covariance
        self.SI = np.zeros((dim_z, dim_z))  # innovation covariance inverse
        self.C = np.zeros((dim_x, dim_z))  # measurement and state cross-covariance

        self._I = np.eye(dim_x)  # identity matrix

        # values of x after prediction step
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        self.Wp = np.ones((1, 2 * dim_x + 1))  # weights for sigma points for state estimation
        self.Wc = np.ones((1, 2 * dim_x + 1))  # weights for sigma points for covariance estimation
        self._update_weights()

        # values of x after update step
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self.inv = np.linalg.inv
        self.cholesky = np.linalg.cholesky
        self.eps = np.eye(dim_x) * 0.0001
        self.max_iter = 10

        self._recalculate_weights = False

    @property  # getter property for alpha
    def alpha(self):
        return self._alpha

    @property  # getter property for alpha
    def beta(self):
        return self._beta

    @property  # getter property for alpha
    def k(self):
        return self._k

    @alpha.setter
    def alpha(self, value):
        if not np.isscalar(value):
            raise ValueError('alpha must be a scalar')
        if value < 1e-4:
            raise ValueError('alpha must be greater than 1e-4')
        if value > 1:
            raise ValueError('alpha must be less than 1')
        self._alpha = value
        self._update_weights()

    @k.setter
    def k(self, value):
        if not np.isscalar(value):
            raise ValueError('k must be a scalar')
        self._k = value
        self._update_weights()

    @beta.setter
    def beta(self, value):
        if not np.isscalar(value):
            raise ValueError('beta must be a scalar')
        self._beta = value
        self._update_weights()

    def _update_weights(self):
        alpha = self.alpha
        beta = self.beta
        k = self.k
        n = self.dim_x

        lamb = (n + k) * alpha ** 2 - n

        Wp = np.ones((1, 2 * n + 1)) / 2 / (n + lamb)
        Wc = Wp.copy()
        Wp[0, 0] *= 2 * lamb
        Wc[0, 0] = Wp[0, 0] + 1 - alpha ** 2 + beta

        self.Wp = Wp
        self.Wc = Wc

    def _safe_cholesky(self):  # This function ensures that P_chol > 0
        eps = self.eps
        max_iter = self.max_iter

        for idx in range(max_iter):
            P_chol = self.cholesky(self.P + idx * eps)
            eig_values = np.linalg.eig(P_chol)

            # if any(any(np.real(eig) < 0) for eig in eig_values):
            #    continue  # we need to add 1 more eps, P_chol must have positive eig values

            return P_chol

        raise ValueError('P_cholesky is ill-conditioned, increase max_iter')

    def _generate_sigma_points(self):

        alpha = self.alpha
        k = self.k

        n = self.dim_x
        lamb = (alpha ** 2) * (n + k) - n
        a = np.sqrt(n + lamb)
        X = np.zeros((n, 2 * n + 1))

        P_chol = a * self._safe_cholesky()
        X[:, 0] = self.x[:, 0]
        X[:, 1:n + 1] = (self.x + P_chol)
        X[:, n + 1:2 * n + 1] = (self.x - P_chol)

        Wc = self.Wc
        Wp = self.Wp

        return X, Wc, Wp

    def prediction(self, u=None, f=None, Q=None):
        if f is None:
            f = self.f
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self.dim_x) * Q

        # First we need to generate 2n + 1 sigma points, n is number of states in the model
        # Wc are weights used for covariance estimation
        # Wp are weights used for mean estimation
        X, Wc, Wp = self._generate_sigma_points()
        # Now we need to propagate the points through the process model
        if u is None:
            X = f(X)
        else:
            if np.isscalar(u):
                u = np.eye(self.dim_u) * u
            X = f(X, u)

        # Now we can use the propagated sigma points for mean and covariance estimation:
        # x = sum(Wp_iX_i)                                 equation for mean calculation
        # P = sum(Wc_i(X_i - x)(X_i - x)')           equation for covariance calculation

        x = (X * Wp).sum(axis=1, keepdims=True)  # Python magic: Wp has the shape (1, 2n+1) and X has shape (n, 2n+1)
        # as a first step Wp*X will result in matrix with shape (n, 2n+1), each row from X will get multiplied
        # element wise with values from row vector Wp, next we want to sum all the columns, that can be achieved
        # by using the np.sum(Wp*X, axis=1), if axis=0 then rows are summed, only issue is that np.sum will return
        # a vector with dimensions (n, ) so we have to use keepdims=True to preserve the dimension (n, 1)

        # More info about broadcasting can be found here: https://numpy.org/doc/stable/user/basics.broadcasting.html

        self.S = ((X - x) * Wc).dot((X - x).T) + Q  # Similarly like before (X-x)*Wp multiples each row of (X-x) with
        # values from row vector Wc.

        self.x = x

        self.x_prior = x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, u=None, h=None, R=None):

        if z is None:  # No measurement is available
            self.x_post = self.x
            self.P_post = self.P
            return

        if h is None:
            h = self.h
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        # First we need to generate 2n + 1 sigma points, n is number of states in the model
        # Wc are weights used for covariance estimation
        # Wp are weights used for mean estimation
        X, Wc, Wp = self._generate_sigma_points()
        x = self.x
        P = self.P

        # Now we need to propagate the points through the measurement model
        if u is None:
            Z = h(X)
        else:
            Z = h(X, u)

        # Now we can use the propagated sigma points for mean and covariance estimation:
        # z_hat = sum(Wp_iX_i)                             equation for mean calculation
        # S = sum(Wc_i(Z_i - z_hat)(Z_i - z_hat)')           equation for covariance calculation
        # C = sum(Wc_i(X_i - x)(Z_i - z_hat)')

        z_hat = np.sum(Z * Wp, axis=1,
                       keepdims=True)  # Python magic: Wp has the shape (1, 2n+1) and X has shape (n, 2n+1)
        # as a first step Wp*X will result in matrix with shape (n, 2n+1), each row from X will get multiplied
        # element wise with values from row vector Wp, next we want to sum all the columns, that can be achieved
        # by using the np.sum(Wp*X, axis=1), if axis=0 then rows are summed, only issue is that np.sum will return
        # a vector with dimensions (n, ) so we have to use keepdims=True to preserve the dimension (n, 1)

        # More info about broadcasting can be found here: https://numpy.org/doc/stable/user/basics.broadcasting.html

        S = ((Z - z_hat) * Wc).dot((Z - z_hat).T) + R  # Similarly like before (X-x)*Wp multiples each row of (X-x) with
        S = S / 2 + S.T / 2
        # values from row vector Wc.
        C = ((X - x) * Wc).dot((Z - z_hat).T)
        SI = self.inv(S)

        # Now we can calculate the Kalman gain:
        # K = C*SI and use it for final update:
        # x = x + K(z - z_hat), v = z - z_hat -> innovation
        # P = P - KSK' - covariance update, to ensure symmetry and numerical stability we will preform
        # correction : P = P/2 + P'/2

        v = z - z_hat
        K = C * SI
        x = x + K * v
        P = P - K * S * K.T
        P = P / 2 + P.T / 2 + self.eps

        self.x = x
        self.v = v
        self.z = z
        self.K = K
        self.S = S
        self.SI = SI
        self.C = C
        self.P = P

        self.x_post = x
        self.P_post = P
