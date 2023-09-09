import numpy as np

import numpy as np
import scipy


def nearPD(A, epsilon=1e-8, zero=0):
    r"""This function clips eigen values of an uncertainty matrix `A`, clipping threshold is specified by `epsilon`.

    Since eigen value represent variances it makes no physical sense for them to be negative, but that can happen
    after the update step in UKF algorthm, to fix this issue we are clipping the eigen values.

    Parameters
    ----------
    A: :class:`~numpy:numpy.ndarray`
        square ``(n, n) ndarray`` representing uncertainty matrix

    epsilon: :py:class:`float`, optional
        lowest possible value for eigen values (default is 0, it is recommended to set it to small positive value)

    Returns
    -------
    A: class:`~numpy:numpy.ndarray`
        reconstructed uncertainty matrix with clipped eigen values.

    Notes
    -----

    An arbitrary square matrix :math:`M` can be decomposed into a form :math:`MV = SV', where :math:'S' is a diagonal
    matrix containing eigen values :math:`\lambda` and :math:`V` is a matrix of eigenvectors. One of the properties
    that the eigenvector matrix satisfies is :math:`VV^T = I`, hence we can reconstruct the original matrix in the
    following way

    .. math::
         M = SVV^T

    TODO : explore a more efficient approach, A v = l v -> (A + cI)v = l v + c v = (l + c) v, i.e, easier to just add
    TODO : 'unit' values to appropriate places, more efficient than direct reconstruction! Issue: this approach doesn't
    TODO : guarantee real values of vector v, but theory implies that if A is symmetric then v and l are real


    """
    C = (A + A.T)/2

    eigval, eigvec = scipy.linalg.eigh(C)

    eigval = np.real(eigval)
    eigvec = np.real(eigvec)

    eigval[eigval < 0] = epsilon*50
    eigval[eigval < zero] = epsilon
    eigval[eigval > 4] = 4

    diag_matrix = np.diag(eigval)

    R = diag_matrix@eigvec@eigvec.T
    R = (R+R.T)/2

    return R

class UnscentedKalmanFilter(object):
    r"""Class used for unscented kalman filter, multiple instances are possible.

    Unscented kalman filter is a filtering algorithm that provides optimal estimates of the states in
    MSE (mean square error) sense, details about the theory can be found in Notes section of this page.

    Attributes
    ----------
    dim_x : :py:class:`int`
        dimensionality of state space, must be 1 or greater

    dim_z : :py:class:`int`
        dimensionality of measurement space, must be 1 or greater

    dim_u : :py:class:`int`
        dimensionality of control space, must be 0 or greater

    x : :class:`~numpy:numpy.ndarray`
        state space array with following shape ``(dim_x, ) ndarray``

    P : :class:`~numpy:numpy.ndarray`
        state space uncertainty matrix with following shape ``(dim_x, dim_x) ndarray``

    P_chol : :class:`~numpy:numpy.ndarray`
        Cholesky decomposition of uncertainty matrix 'P' with following shape ``(dim_x, dim_x) ndarray``

    x_prior : :class:`~numpy:numpy.ndarray`
        hard copy of the state 'x' after the predict step, shape ``(dim_x, ) ndarray``

    P_prior : :class:`~numpy:numpy.ndarray`
        hard copy of the matrix 'P' after the predict step, shape ``(dim_x, dim_x) ndarray``

    x_post : :class:`~numpy:numpy.ndarray`
        hard copy of the state 'x' after the update step, shape ``(dim_x, ) ndarray``

    P_post : :class:`~numpy:numpy.ndarray`
        hard copy of the matrix 'P' after the update step, shape ``(dim_x, dim_x) ndarray``

    X_norm : :class:`~numpy:numpy.ndarray`, optional
        matrix used for normalisation of X sigma points, shape ``(dim_x, dim_x) ndarray``.

    X_renorm : :class:`~numpy:numpy.ndarray`, optional
        matrix used for renormalization of X sigma points, shape ``(dim_x, dim_x) ndarray``.

    Z_norm : :class:`~numpy:numpy.ndarray`, optional
        matrix used for normalisation of Z sigma points, shape ``(dim_x, dim_x) ndarray``.

    Z_renorm : :class:`~numpy:numpy.ndarray`, optional
        matrix used for renormalization of Z sigma points, shape ``(dim_x, dim_x) ndarray``.

    divergence_uncertainty : :class:`~numpy:numpy.ndarray`
        matrix that is used as the value of 'P_chol' in the case that
        :func:`~unscented_kalman_filter.UnscentedKalmanFilter._safe_cholesky` fails to find the Cholesky decomposition

    Q : :class:`~numpy:numpy.ndarray`
        matrix representing uncertainty in estimator model with following shape ``(dim_x, dim_x) ndarray``

    f : function
        User specified function that is used to predict the future state of the model

    z : :class:`~numpy:numpy.ndarray`
        measurement space array with following shape ``(dim_z, ) ndarray``

    R : :class:`~numpy:numpy.ndarray`
        matrix representing uncertainty in measurements ``(dim_z, dim_z) ndarray``

    K : :class:`~numpy:numpy.ndarray`
        Kalman gain matrix with following shape ``(dim_x, dim_z) ndarray``

    v : :class:`~numpy:numpy.ndarray`
        innovation array with following shape ``(dim_z, ) ndarray``, basically error between predicted measurement and
        true measurement, it can be used to verify the filter - if the filter is properly tuned `v` should have
        statistical properties of white noise.

    S : :class:`~numpy:numpy.ndarray`
        matrix representing uncertainty in the innovation array, shape ``(dim_z, dim_z) ndarray``

    SI : :class:`~numpy:numpy.ndarray`
        inverse of `S`, calculated once to reduce the number of inversions

    C : :class:`~numpy:numpy.ndarray`
        matrix representing cross-correlation between state and measurement space, shape ``(dim_x, dim_z) ndarray``

    Wp : :class:`~numpy:numpy.ndarray`
        weights for sigma points mean estimation, shape ``(2 * dim_x + 1, 1) ndarray``. This should not be specified
        by the user.

    Wc : :class:`~numpy:numpy.ndarray`
        weights for sigma points covariance estimation, shape ``(2 * dim_x + 1, 1) ndarray``. This should not be
        specified by the user.

    eps_Q: :class:`~numpy:numpy.ndarray`
        Matrix filled with small values used to guarantee that all eigen values of P are > 0,
        this can be specified by the user, shape ``(dim_x, dim_x) ndarray``

    eps_R: :class:`~numpy:numpy.ndarray`
        Matrix filled with small values used to guarantee that all eigen values of S are > 0,
        this can be specified by the user, shape ``(dim_z, dim_z) ndarray``

    Methods
    -------
    prediction(u=None, f=None, Q=None, **kwargsf):
        Starts the prediction step of filtering algorithm, after the prediction step is done results are stored in
        `x_prior` and `P_prior`
    update(self, z, h=None, R=None, **kwargsh):
        Starts the update step of filtering algorithm, after the prediction step is done results are stored in
        `x_post` and `P_post`

    Notes
    -----

    Private attributes `_alpha`, `_beta` and `_k` are used to determine the spread of sigma points. Spread can be
    decreased by decreasing `_alpha` or `_k`. Spread is very sensitive to values of `_alpha`, fine-tuning can be
    achieved by changing the value of `_k`. After the values `_alpha`, `_beta` and `_k` have been changed
    (that can be achieved by using setter functions, i.e., `ukf.alpha = value`) weights `Wc` and `Wp` get updated
    automatically. It is recommended to not decrease `_alpha` more than 1e-4 because numerical issued arise when
    small values of `_alpha` are used.

    In usual use case it is recommended to use only `x_post`, `x_prior`, `P_post` and `P_prior` as the outputs of
    the filter, because only these variables are hard copies, other variables might get change via the reference if
    a hard copy is not made when they are extracted from unscented kalman filter.

    `dt_estimation`, `timing_estimation` are optional attributes that can be used externally for the purpose of
    synchronisation, internally they have no use. `dt_kalman_gain` and `timing_kalman_gain` can be used to
    change the rate at which is kalman gain updated.

    If `X_norm` and `Z_norm` normalisation matrices are used it is assumed that internally the states and uncertainty
    matrices are normalised - only when we are using the estimation/measurement model we are re-normalising the sigma
    points.

    """

    def __init__(self, dim_x=1, dim_z=1, dim_u=0):
        r""" Constructor for the :class:`UnscentedKalmanFilter`

        Parameters
        ----------
        dim_x: :py:class:`int`
            dimension of the state space, must be 1 or greater
        dim_z: :py:class:`int`
            dimension of the measurement space, must be 1 or greater
        dim_u: :py:class:`int`
            dimension of the control space, must be 0 or greater

        Notes
        -----
        Other attributes of the :class:`UnscentedKalmanFilter` are set to default values, before any use
        user must specify those attributes by accessing the object fields.

        """

        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')



        # values used for the purposes of synchronisation, estimation gets performed only if t - timing > dt, otherwise
        # appropriate prediction value gets returned, timing is incremented by dt each time the estimation is pref.
        self.dt_estimation = 1
        self.timing_estimation = 0

        self.dt_kalman_gain = 1
        self.timing_kalman_gain = 0

        # Names of the variables that can be specified by the user, it makes it easier to plot out the data later on
        # names should be formatted as a dictionary where the key is position in the state array, and value is the name
        # of the variable, currently unused, might use them in the future for plotting purposes
        self.state_names = None
        self.measurement_names = None
        self.control_names = None

        # dimensionality of state space, measurement space and control space
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        # Parameters for sigma point generation :
        self._alpha = 1
        self._beta = 0
        self._k = 3 - self.dim_x

        self.x = np.zeros((dim_x,))  # state
        self.P = np.eye(dim_x)  # estimation uncertainty covariance
        self.P_chol = np.eye(dim_x)  # Cholesky decomposition of P
        self.Q = np.eye(dim_x)  # process uncertainty
        self.f = None  # non-linear process function

        self.z = np.zeros((dim_z,))  # measurement
        self.h = None  # non-linear measurement function
        self.R = np.eye(dim_z)  # measurement uncertainty

        self.K = np.zeros((dim_x, dim_z))  # Kalman gain matrix
        self.v = np.zeros((dim_z,))  # innovation vector
        self.S = np.zeros((dim_z, dim_z))  # innovation covariance
        self.SI = np.zeros((dim_z, dim_z))  # innovation covariance inverse
        self.C = np.zeros((dim_x, dim_z))  # measurement and state cross-covariance

        self._I = np.eye(dim_x)  # identity matrix

        # values of x after prediction step
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        self.Wp = np.ones((2 * dim_x + 1, 1))  # weights for sigma points for state estimation
        self.Wc = np.ones((2 * dim_x + 1, 1))  # weights for sigma points for covariance estimation
        self._update_weights()

        # values of x after update step
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        self.inv = np.linalg.inv  # storing the linalg.inv function for easier access
        self.cholesky = np.linalg.cholesky  # storing the linalg.cholesky function for easier access
        self.eps_Q = np.eye(dim_x) * 0  # Can be changed, used to guarantee that all eigen values of P are > 0
        self.eps_R = np.eye(dim_z) * 0  # Can be changed, used to guarantee that all eigen values of P are > 0
        self.divergence_uncertainty = 5   # Can be changed, represents our uncertainty in estimation after algorithm
        # diverges

    @property  # getter property for alpha
    def alpha(self):
        """Getter and setter methods for `_alpha` are defined using the
        property tag, value of `_alpha` can be got/set using the regular syntax for
        attribute access
        """
        return self._alpha

    @property  # getter property for alpha
    def beta(self):
        """Getter and setter method for _beta are defined using the
        property tag, value of `_beta` can be got/set using the regular syntax for
        attribute access
        """
        return self._beta

    @property  # getter property for alpha
    def k(self):
        """Getter and setter method for _k are defined using the
        property tag, value of `_k` can be got/set using the regular syntax for
        attribute access
        """
        return self._k

    @alpha.setter
    def alpha(self, value):
        # alpha parameter determines the spread of sigma points
        # small values of alpha increase the accuracy of estimation
        # but that leads to larger value of weights and more numerical errors
        # alpha should never be larger than 1, that is a theoretical limit.
        if not np.isscalar(value):
            raise ValueError('alpha must be a scalar')
        if value < 1e-4:
            raise ValueError('alpha must be greater than 1e-4')
        if value > 1:
            raise ValueError('alpha must be less than 1')
        self._alpha = value

        # now we update the value of weights, they depend on alpha, beta and k
        self._update_weights()

    @k.setter
    def k(self, value):
        # parameter that also impacts the spread of sigma points, it's recommended tuning k over alpha
        # since weight values are less sensitive to k
        # increasing k increases the spread and decreasing reduces it
        # only real limit is that k must be greater than -dim_x, because at some point we calculate the value
        # of expression sqrt(dim_x + k)

        if not np.isscalar(value):
            raise ValueError('k must be a scalar')
        if value <= -self.dim_x:
            raise ValueError('k must greater than -x')
        self._k = value

        # now we update the value of weights, they depend on alpha, beta and k
        self._update_weights()

    @beta.setter
    def beta(self, value):
        # this parameter impacts the estimation of distribution skewness, for Gaussian distributions
        # beta = 2 is an optimal choice, deviating from this choice can increase the accuracy if we
        # know in which direction is the transformed distribution skewed

        if not np.isscalar(value):
            raise ValueError('beta must be a scalar')
        self._beta = value

        # now we update the value of weights, they depend on alpha, beta and k
        self._update_weights()

    def _update_weights(self):
        r""" This function updates the value of weights Wp and Wc.

        Notes
        -----
        Calculating the weight values using the theoretical expressions from Sigma_Point_Kalman_filters(UKF+PF)
        paper in NoNLin Filtering section of the GDrive, it is not recommended to change these expressions
        unless there is some theoretical justification
        """

        # Reading the parameter values for the purpose of having more readable mathematical expressions
        alpha = self.alpha
        k = self.k
        n = self.dim_x
        beta = self.beta
        lamb = (n + k) * alpha ** 2 - n

        # Calculating the weights
        Wp = self.Wp * 0 + 1
        Wp /= 2 * (n + lamb)
        Wc = self.Wc * 0 + 1
        Wc /= 2 * (n + lamb)
        Wp[0, 0] *= 2 * lamb
        Wc[0, 0] = Wp[0, 0] + 1 - alpha ** 2 + beta
        #Wc[0, 0] = 0  # setting this to 0 guarantees that predicted matrix P > 0

        # Storing the calculated values
        self.Wp = Wp
        self.Wc = Wc

    def _safe_cholesky(self):
        r""" This is used to find the Cholesky decomposition of matrix `P` in a safe way.

        This is a private method, it is not part of the public API.

        Notes
        -----
        Cholesky decomposition tries to find a 'square-root' of a matrix, i.e., a matrix that satisfies the following
        expression: :math:`P = AA^T` , for the procedure to work matrix P must satisfy following conditions:
        * `P` is a symmetric matrix, :math:`P = P^T`
        * `P` > 0, i.e. `P` must not have any eigen value that is < 0

        Theoretically we don't expect `P` to have any eig values < 0, since that implies complex value
        of standard deviation, so if it happens that `P` < 0 that is a result of either alg diverging or numerical err,
        we are fixing that by utilising :func:`near_PD` function, essentially it clips all the eigen values from some
        epsilon.

        In case that the procedure fails we set the value of 'P_chol' to 'divergence_uncertainty'.
        """

        # Reading the values

        P = self.P

        self.P = nearPD(P)

        try:
            self.P_chol = self.cholesky(self.P)
        except:
            self.P_chol = self.divergence_uncertainty*self.Q

    def _generate_sigma_points(self):
        r"""This function generates sigma points for purposes of inference of statistical
            properties of non-linear process model.

            This is a private method, it is not part of the public API.

            Notes
            -----
            For the purpose of Sigma point generation theoretical expressions from Sigma_Point_Kalman_filters(UKF+PF)
            paper in NoNLin Filtering section of the GDrive are being used, it is not recommended to change these
            expressions unless there is some theoretical justification.

            Central point of generated Sigma points is around current estimate of `x`, 2n points are spread
            around it, the spread is determined by uncertainty matrix `P` and choice of parameters `_alpha` and `_k`,
            if the spread of points is smaller we will get more accurate image of nonlinear trans.

        """

        # Reading the parameter values for the purpose of having more readable mathematical expressions
        alpha = self.alpha
        k = self.k

        n = self.dim_x
        lamb = (alpha ** 2) * (n + k) - n
        a = np.sqrt(n + lamb)

        X = np.zeros((2 * n + 1, n))
        self._safe_cholesky()  # updating self.P_chol
        P_chol = a * self.P_chol

        # calculating the sigma points
        X[0, :] = self.x
        X[1:n + 1, :] = (self.x + P_chol)
        X[n + 1:2 * n + 1, :] = (self.x - P_chol)

        return X

    def prediction(self, u=None, f=None, Q=None, **kwargsf):
        r"""This is the prediction step of filtering algorithm.

        Based on the current estimation of state space `x`, current values of control `u` and estimator model `f`
        we predict the state of the system 1 step in the future.

        Parameters
        ----------
        u: :class:`~numpy:numpy.ndarray`, optional
            Control signals at the current time step (default value is None, if `u` is None then it's not passed to
            estimator model `f`).

        f : function, optional
            Estimator model function (default is None, if `f` is None then function stored in attribute `self.f` is
            used as estimator model)

        Q : :class:`~numpy:numpy.ndarray`, optional
            Process uncertainty matrix (default is None, if `Q` is None then matrix stored in attribute `self.Q` is
            used as process uncertainty)

        **kwargsf : :py:class:`dict`, optional
            dictionary of keyword arguments that get passed along to estimator model `f` (default is None)

        Notes
        -----
        Results of the update step are stored in `x_prior` and `P_prior` these variables are hard copies, so they can
        be used safely outside the :class:`UnscentedKalmanFilter`

        """
        # This is the prediction step of Kalman algorithm, based on the current estimation of state space, current
        # values of control u and process model f we predict the state of the system 1 step in the future

        # Just making the function more flexible, in usual use case only u will get passed in as an argument,
        # but there might exist some use cases where f and Q could be adaptive based on some external logic
        if f is None:
            f = self.f
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self.dim_x) * Q

        # First we need to generate 2n + 1 sigma points, n is number of states in the model
        # Wc are weights used for covariance estimation
        # Wp are weights used for mean estimation
        X = self._generate_sigma_points()

        # Now we need to propagate the points through the process model
        if u is None:
            X = f(X.reshape((*X.shape, 1)),
                              **kwargsf)  # need to add 1 extra dimension so the data gets treated as N 1-d inputs
            if X.ndim == 3:
                X = X.squeeze(axis=2)
        else:
            if np.isscalar(u):
                u = np.array([u])  # need to add 1 extra dimension so the data gets treated as N 1-d inputs
            # 1 extra dimension is added to u as well, because by the default numpy expands from the left
            # and that would lead to errors, i.e., (2,) -> (1, 2)
            X = f(X.reshape((*X.shape, 1)),
                              u.reshape((*u.shape, 1)), **kwargsf)
            if X.ndim == 3:
                X = X.squeeze(axis=2)

        # Now we can use the propagated sigma points for mean and covariance estimation:
        # x = sum(Wp_iX_i)                                 equation for mean calculation
        # P = sum(Wc_i(X_i - x)(X_i - x)')           equation for covariance calculation
        # Wp and Wc weights add up to 1, i.e., sum(Wp) = sum(Wc) = 1

        x = (X * self.Wp).sum(axis=0)  # Python magic: Wp has the shape (2n+1, 1) and X has shape (2n+1, n)
        # as a first step Wp*X will result in matrix with shape (2n+1, n), each row from X will get multiplied
        # element wise with values from row vector Wp, next we want to sum all the rows, that can be achieved
        # by using the np.sum(Wp*X, axis=0), if axis=1 then columns are summed

        # More info about broadcasting can be found here: https://numpy.org/doc/stable/user/basics.broadcasting.html

        # This expression estimates the uncertainty in the predicted position of the x
        # using the standard expression for covariance estimation
        self.P = ((X - x) * self.Wc).T @ (X - x) + Q  # Similarly like before (X-x)*Wp
        # multiples each row of (X-x) with values from row vector Wc.

        self.P = self.P / 2 + self.P.T / 2

        self.x = x

        self.x_prior = x.copy()
        self.P_prior = self.P.copy()

    def update(self, z, h=None, R=None, update_gain=True, **kwargsh):
        r"""This is the update step of filtering algorithm

            Based on our confidence in the model and measurements we make optimal decision - if we trust the model more
            than the measurement then the estimation is more biased towards the predicted value `x`,
            in other case we are biased towards the measurement `z`.

            Parameters
            ----------
            z: :class:`~numpy:numpy.ndarray`
                Measurement at the current time step (if `z` is None it is assumed that no measurement is available,
                in that case we use predicted value as the estimate).

            h : function, optional
                Measurement model function (default is None, if `h` is None then function stored in attribute `self.h`
                is used as measurement model)

            R : :class:`~numpy:numpy.ndarray`, optional
                Measurement uncertainty matrix (default is None, if `Q` is None then matrix stored in attribute `self.R`
                is used as measurement uncertainty)

            update_gain : :py:class`bool`, optional
                used to specify if the Kalman gain should be updated (default value is ``True``)

            **kwargsh : :py:class:`dict`, optional
                dictionary of keyword arguments that get passed along to measurement model `h` (default is None)

            Notes
            -----
            Results of the update step are stored in `x_post` and `P_post` these variables are hard copies, so they can
            be used safely outside the :class:`UnscentedKalmanFilter`

        """
        # This is the update step of Kalman algorithm, based on our confidence in the model and measurements
        # we make optimal decision - if we trust the model more than the measurement then the estimation is more
        # biased with state space values more consistent with the model, in other case we are biased towards the
        # measurement

        # z is the measurement at the present time

        if z is None:  # No measurement is available, value of P could be restarted to reflect our lack
            # of confidence in predicted values, since no measurement is available
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.z = np.array([None] * self.dim_z)
            self.v = np.array(0 * self.dim_z)
            return
        else:
            z =  z
        #  Just making the function more flexible, in usual use case only u will get passed in as an argument,
        #  but there might exist some use cases where h and R could be adaptive based on some external logic
        if h is None:
            h = self.h
        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        # First we need to generate 2n + 1 sigma points, n is number of states in the model
        # Wc are weights used for covariance estimation
        # Wp are weights used for mean estimation
        X = self._generate_sigma_points()
        x = self.x
        P = self.P

        # Now we need to propagate the points through the measurement model
        # X here contains n columns and 2n+1 rows. Each row is a state prediction is the state space for a given sigma
        # point. The following instruction unpacks the rows (each associated to a sigma point) so that each sigma point
        # prediction becomes an element of an array instead of a row of a matrix.
        # it also applies the measurement model, i.e. takes predicted state X to produce predicted measurement Z
        Z = h(X.reshape((*X.shape, 1)),
                          **kwargsh)  # need to add 1 extra dimension so the data gets treated as N 1-d inputs
        if Z.ndim == 3:  # bring Z back to be a 2D array of array, for easier handling
            Z = Z
            Z = Z.squeeze(axis=2)

        # Now we can use the propagated sigma points for mean and covariance estimation:
        # z_hat = sum(Wp_iX_i)                             equation for mean calculation
        # S = sum(Wc_i(Z_i - z_hat)(Z_i - z_hat)')         equation for covariance calculation
        # C = sum(Wc_i(X_i - x)(Z_i - z_hat)')             equation for cross-covariance
        # Wp and Wc weights add up to 1, i.e., sum(Wp) = sum(Wc) = 1

        # z_hat is the center/the best prediction among the scattered Z points
        z_hat = np.sum(Z * self.Wp, axis=0)  # Python magic: Wp has the shape (2n+1, 1) and X has shape (2n+1, n)
        # as a first step Wp*X will result in matrix with shape (2n+1, n), each row from X will get multiplied
        # element wise with values from row vector Wp, next we want to sum all the rows, that can be achieved
        # by using the np.sum(Wp*X, axis=0), if axis=1 then columns are summed

        # More info about broadcasting can be found here: https://numpy.org/doc/stable/user/basics.broadcasting.html

        # This expression estimates the uncertainty in the measurement z using
        # the standard expression for covariance estimation
        S = ((Z - z_hat) * self.Wc).T @ (Z - z_hat) + R  # Similarly like before (X-x)*Wp
        # multiples each row of (X-x) with values from row vector Wc.

        S = S / 2 + S.T / 2  # Making sure that S is a symmetric matrix

        # This expression estimates the similarity between current state and measurement
        C = ((X - x) * self.Wc).T @ (Z - z_hat)  # each row of (X-x) is multiplied by a scalar, element of array Wc
        # (X - x) * self.Wc).T has dimension [n 2n+1] , (Z - z_hat) has dim [2n+1,size of meas array]
        # C has dim [n, size of meas array

        if update_gain:
            SI = self.inv(S)  # size of S matrix depends on the size of measurements array
            K = C @ SI  # kalman gain, decides what to trust more, measurements or model
        else:
            K = self.K
            SI = self.SI

        # Now we can calculate the Kalman gain:
        # K = C*SI and use it for final update:
        # x = x + K(z - z_hat), v = z - z_hat -> innovation
        # P = P - KSK' - covariance update, to ensure symmetry and numerical stability we will preform
        # correction : P = P/2 + P'/2

        # high uncertainty: big S, small SI, small P (P~C)

        v = z - z_hat  # innovation info element, difference between actual observation and predicted observation
        x = x + K @ v  # update the final estimate of state array with the innovation information
        P = P - K @ S @ K.T  # update the covariance matrix (uncertainty in the state space) based on kalman gain
        P = P / 2 + P.T / 2  # Making sure that P is a symmetric matrix

        self.x = x
        self.v = v
        self.z = z
        self.K = K
        self.S = S
        self.SI = SI
        self.C = C
        self.P = P
        self.x_post = x.copy()
        self.P_post = P.copy()




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

        self.x = np.zeros((dim_x, ))  # state
        self.x_info = np.zeros((dim_x, ))  # state in information space
        self.P = np.eye(dim_x)  # estimation uncertainty covariance
        self.P_chol = np.eye(dim_x)  # Cholesky decomposition of P
        self.P_inv = np.eye(dim_x)  # estimation uncertainty covariance inversion
        self.Q = np.eye(dim_x)  # process uncertainty
        self.f = None  # non-linear process function

        self.z = np.zeros((dim_z, ))  # measurement
        self.h = None  # non-linear measurement function
        self.R_inv = np.eye(dim_z)  # measurement uncertainty

        self.K = np.zeros((dim_x, dim_z))  # Kalman gain matrix
        self.v = np.zeros((dim_z, 1))  # innovation vector
        self.S = np.zeros((dim_z, dim_z))  # innovation covariance
        self.C = np.zeros((dim_x, dim_z))  # measurement and state cross-covariance

        self._I = np.eye(dim_x)  # identity matrix

        # values of x after prediction step
        self.x_prior = self.x.copy()
        self.P_inv_prior = self.P.copy()

        self.Wp = np.ones((2 * dim_x + 1, 1))  # weights for sigma points for state estimation
        self.Wc = np.ones((2 * dim_x + 1, 1))  # weights for sigma points for covariance estimation
        self._update_weights()

        # values of x after update step
        self.x_post = self.x.copy()
        self.P_inv_post = self.P.copy()

        self.inv = np.linalg.inv
        self.cholesky = np.linalg.cholesky
        self.eps = np.eye(dim_x) * 0.0001
        self.max_iter = 10
        self.divergence_uncertainty = 5

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
        Wp = self.Wp * 0 + 1
        Wp /= 2 * (n + lamb)
        Wc = self.Wc * 0 + 1
        Wc /= 2 * (n + lamb)
        Wp[0, 0] *= 2 * lamb
        Wc[0, 0] = Wp[0, 0] + 1 - alpha ** 2 + beta

        self.Wp = Wp
        self.Wc = Wc

    def _safe_cholesky(self):
        r""" This is used to find the Cholesky decomposition of matrix `P` in a safe way.

        This is a private method, it is not part of the public API.

        Notes
        -----
        Cholesky decomposition tries to find a 'square-root' of a matrix, i.e., a matrix that satisfies the following
        expression: :math:`P = AA^T` , for the procedure to work matrix P must satisfy following conditions:
        * `P` is a symmetric matrix, :math:`P = P^T`
        * `P` > 0, i.e. `P` must not have any eigen value that is < 0

        Theoretically we don't expect `P` to have any eig values < 0, since that implies complex value
        of standard deviation, so if it happens that `P` < 0 that is a result of either alg diverging or numerical err,
        we are fixing that by utilising :func:`near_PD` function, essentially it clips all the eigen values from some
        epsilon.

        In case that the procedure fails we set the value of 'P_chol' to 'divergence_uncertainty'.
        """

        # Reading the values

        P = self.P

        self.P = nearPD(P)

        try:
            self.P_chol = self.cholesky(self.P)
        except:
            self.P_chol = self.divergence_uncertainty*self.Q

    def _generate_sigma_points(self):

        alpha = self.alpha
        k = self.k

        n = self.dim_x
        lamb = (alpha ** 2) * (n + k) - n
        a = np.sqrt(n + lamb)
        X = np.zeros((2 * n + 1, n))
        self._safe_cholesky()
        P_chol = a * self.P_chol
        X[0, :] = self.x
        X[1:n + 1, :] = (self.x + P_chol)
        X[n + 1:2 * n + 1, :] = (self.x - P_chol)

        return X

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
        X = self._generate_sigma_points()
        # Now we need to propagate the points through the process model
        if u is None:
            X = f(X.reshape((*X.shape, 1)))  # need to add 1 extra dimension so the data gets treated as N 1-d inputs
            if X.ndim == 3:
                X = X.squeeze(axis=2)
        else:
            if np.isscalar(u):
                u = np.array([u])

            X = f(X.reshape((*X.shape, 1)), u.reshape((*u.shape, 1)))  # need to add 1 extra dimension so the data gets
            # treated as N 1-d inputs
            if X.ndim == 3:
                X = X.squeeze(axis=2)

        # Now we can use the propagated sigma points for mean and covariance estimation:
        # x = sum(Wp_iX_i)                                 equation for mean calculation
        # P = sum(Wc_i(X_i - x)(X_i - x)')           equation for covariance calculation

        x = (X * self.Wp).sum(axis=0)  # Python magic: Wp has the shape (2n+1, 1) and X has shape (2n+1, n)
        # as a first step Wp*X will result in matrix with shape (2n+1, n), each row from X will get multiplied
        # element wise with values from row vector Wp, next we want to sum all the rows, that can be achieved
        # by using the np.sum(Wp*X, axis=0), if axis=1 then columns are summed

        # More info about broadcasting can be found here: https://numpy.org/doc/stable/user/basics.broadcasting.html
        self.P = ((X - x) * self.Wc).T @ (X - x) + Q
        self.P_inv = self.inv(self.P)

        self.x = x
        self.x_info = self.P_inv @ self.x

        self.x_prior = x.copy()
        self.P_inv_prior = self.P_inv.copy()

    def update(self, z, h=None, R_inv=None, multiple_sensors=False):

        if z is None:  # No measurement is available
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.z = np.array([None] * self.dim_z)
            self.v = np.array(0 * self.dim_z)
            return

        if h is None:
            h = self.h
        if R_inv is None:
            R_inv = self.R_inv
        elif np.isscalar(R_inv):
            R_inv = np.eye(self.dim_z) * R_inv

        # First we need to generate 2n + 1 sigma points, n is number of states in the model
        # Wc are weights used for covariance estimation
        # Wp are weights used for mean estimation
        X = self._generate_sigma_points()
        x = self.x
        P = self.P

        # Now we need to propagate the points through the measurement model

        if multiple_sensors:
            number_of_sensors = z.shape[0]   # It is assumed that measurements are stacked in rows
            self.v = np.zeros((number_of_sensors, self.dim_z))
            C = np.zeros((number_of_sensors, self.dim_x, self.dim_z))
            bias = np.zeros((number_of_sensors, self.dim_z))
            for idx in range(number_of_sensors):
                Z = h[idx](X.reshape((*X.shape, 1)))
                if Z.ndim == 3:
                    Z = Z.squeeze(axis=2)

                z_hat = np.sum(Z * self.Wp, axis=0)
                self.v[idx, :] = z[idx, :] - z_hat

                C[idx, :, :] = ((X - x) * self.Wc).T @ (Z - z_hat)
                bias[idx, :] = C[idx, :, :].T @ self.x_info

            self.v = self.v.reshape((*self.v.shape, 1))
            bias = bias.reshape((*bias.shape, 1))
        else:
            number_of_sensors = 1  # It is assumed that measurements are stacked in rows
            self.v = np.zeros((number_of_sensors, self.dim_z))
            C = np.zeros((number_of_sensors, self.dim_x, self.dim_z))
            bias = np.zeros((number_of_sensors, self.dim_z))
            for idx in range(number_of_sensors):
                Z = h(X.reshape((*X.shape, 1)))
                if Z.ndim == 3:
                    Z = Z.squeeze(axis=2)

                z_hat = np.sum(Z * self.Wp, axis=0)
                self.v[idx, :] = z - z_hat

                C[idx, :, :] = ((X - x) * self.Wc).T @ (Z - z_hat)
                bias[idx, :] = C[idx, :, :].T @ self.x_info

            self.v = self.v.reshape((*self.v.shape, 1))
            bias = bias.reshape((*bias.shape, 1))
            R_inv = R_inv.reshape((1, *R_inv.shape))

        C_T = C.transpose(0, 2, 1)  # Transposing only the matrices
        ik = (self.P_inv @ C @ R_inv @ (self.v + bias)).sum(axis=0)  # sensor information contribution
        ik = ik.squeeze()
        Ik = (self.P_inv @ C @ R_inv @ C_T @ self.P_inv.T).sum(axis=0)  # sensor uncertainty contribution

        self.x_info += ik
        self.P_inv += Ik
        self.P = self.inv(self.P_inv)
        self.P = self.P / 2 + self.P.T / 2 + self.eps

        self.x = self.P @ self.x_info
        #self.v = v
        self.z = z
        self.C = C
        self.P = P

        self.x_post = x.copy()
        self.P_inv_post = P.copy()
