import numpy as np

import matplotlib.pyplot as plt
#%pylab inline
#%config InlineBackend.figure_format = 'retina'

import seaborn as sns

import tensorflow as tf

np.random.seed(0)
import h5py
import scipy.io

import scipy
import scipy.stats as stats

dtype = tf.float64
dtype_convert = tf.to_double

import time


def normalise_data(X, shouldNormalise):
    if shouldNormalise:
        X_mean = np.mean(X, axis=0)
        X = X - X_mean
        X_std = X.std(axis=0, ddof=1, keepdims=True)
        X_std[X_std < 1e-10] = 1
        X_std = np.maximum(X_std, 1e-8)
        X = X / X_std
        assert (np.all(X != np.NaN))
    else:
        X_mean = 0
        X_std = 1

    return X, X_mean, X_std


def standard_pca_initialisation(Y, Q):
    assert (Y.shape[0] >= Q)
    assert (Y.shape[1] >= Q)

    Y_norm, Y_mean, Y_std = normalise_data(Y, shouldNormalise=True)

    if Y.shape[0] < (Q + 2):
        W, V = scipy.linalg.eig(np.dot(Y_norm, Y_norm.T))
        V = V[:, 0:Q]
    else:
        W, V = scipy.sparse.linalg.eigs(np.dot(Y_norm, Y_norm.T), Q)

    X0 = np.real(V)
    X0 = X0 / np.mean(X0.std(axis=0, ddof=1))

    return X0


def log_det_from_chol(L):
    return 2.0 * tf.reduce_sum(tf.log(tf.diag_part(L)))


def real_variable(inital_value, prior_function=None, name=None):
    t_a = tf.Variable(inital_value, dtype=dtype, name=name)
    return t_a


def positive_variable(inital_value, prior_function=None, name=None):
    t_a = tf.Variable(tf.log(tf.cast(inital_value, dtype=dtype)), dtype=dtype, name=name)
    return tf.exp(t_a)


# Will ensure a monotonic sequence in the range [-1,1] without the end points hard set to -1 and 1.
def monotonic_sequence_variable(N_samples, init, min=-1.0, max=1.0):
    assert issubclass(type(init), np.ndarray)
    s = list(np.shape(init))
    s[0] = 1
    init = np.concatenate([-10.0 * np.ones(s), init])
    t_raw = real_variable(init)
    t_sm = tf.nn.softmax(t_raw, dim=0)
    t_seq = (max - min) * tf.cumsum(t_sm, axis=0, exclusive=True) + min
    t_seq = t_seq[1:, ]
    return t_seq, t_raw


class Kernel:
    def __init__(self, covar_matrix_func, covar_diag_func, descriptor, kernels=None):
        self._descriptor = descriptor
        self._covar_matrix_func = covar_matrix_func
        self._covar_diag_func = covar_diag_func
        self._kernels = kernels

    @property
    def descriptor(self):
        return self._descriptor

    def covar_matrix(self, t_X, t_Z):
        return self._covar_matrix_func(t_X=t_X, t_Z=t_Z)

    def covar_diag(self, t_X):
        return self._covar_diag_func(t_X=t_X)


def create_sum_kernel(kernels): # e.g. kernels=[create_squared_exp_kernel(..), create_matern32_kernel(..)]
    assert all([issubclass(type(k), Kernel) for k in kernels])

    def matrix_func(t_X, t_Z):
        return tf.reduce_sum(tf.stack([k.covar_matrix(t_X, t_Z) for k in kernels]), axis=0)

    def diag_func(t_X):
        return tf.reduce_sum(tf.stack([k.covar_diag(t_X) for k in kernels]), axis=0)

    descriptor = 'Sum (' + ', '.join([k.descriptor for k in kernels]) + ')'
    return Kernel(descriptor=descriptor,
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=kernels)

def create_squared_exp_kernel(t_alpha, t_gamma):
    def matrix_func(t_X, t_Z):
        xx = 0.5 * t_gamma * tf.reduce_sum(t_X * t_X, axis=1, keep_dims=True)
        zz = 0.5 * t_gamma * tf.reduce_sum(t_Z * t_Z, axis=1, keep_dims=True)
        sq_dist_xz = xx + tf.transpose(zz) - t_gamma * tf.matmul(t_X, t_Z, transpose_b=True)
        return t_alpha * tf.exp(- sq_dist_xz)

    def diag_func(t_X):
        t_N = tf.shape(t_X)[0]
        return t_alpha * tf.ones([t_N], dtype=dtype)

    return Kernel(descriptor='Squared Exponential',
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=None)

def create_periodic_kernel(t_alpha, t_gamma, t_period): # inverse period
    def matrix_func(t_X, t_Z):
        xx = tf.reduce_sum(t_X * t_X, axis=1, keep_dims=True)
        zz = tf.reduce_sum(t_Z * t_Z, axis=1, keep_dims=True)
        dist_xz = xx + tf.transpose(zz) - 2.0 * tf.matmul(t_X, t_Z, transpose_b=True)
        return t_alpha * tf.exp(- 2.0 * t_gamma * (tf.sin(t_period * tf.sqrt(dist_xz + 1.0e-12)) ** 2))

    def diag_func(t_X):
        t_N = tf.shape(t_X)[0]
        return t_alpha * tf.ones([t_N], dtype=dtype)

    return Kernel(descriptor='Periodic',
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=None)


def create_matern32_kernel(t_alpha, t_gamma):
    def matrix_func(t_X, t_Z):
        xx = 0.5 * t_gamma * tf.reduce_sum(t_X * t_X, axis=1, keep_dims=True)
        zz = 0.5 * t_gamma * tf.reduce_sum(t_Z * t_Z, axis=1, keep_dims=True)
        sq_dist_xz = xx + tf.transpose(zz) - t_gamma * tf.matmul(t_X, t_Z, transpose_b=True)
        sqrt_3 = tf.constant(np.sqrt(3.0), dtype=dtype)
        sqrt_3_dist_xz = sqrt_3 * tf.sqrt(sq_dist_xz + 1.0e-12)
        return t_alpha * (1.0 + sqrt_3_dist_xz) * tf.exp(- sqrt_3_dist_xz)

    def diag_func(t_X):
        t_N = tf.shape(t_X)[0]
        return t_alpha * tf.ones([t_N], dtype=dtype)

    return Kernel(descriptor='Matern 3/2',
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=None)


def create_matern12_kernel(t_alpha, t_gamma):
    def matrix_func(t_X, t_Z):
        xx = 0.5 * t_gamma * tf.reduce_sum(t_X * t_X, axis=1, keep_dims=True)
        zz = 0.5 * t_gamma * tf.reduce_sum(t_Z * t_Z, axis=1, keep_dims=True)
        sq_dist_xz = xx + tf.transpose(zz) - t_gamma * tf.matmul(t_X, t_Z, transpose_b=True)
        dist_xz = tf.sqrt(sq_dist_xz + 1.0e-12)
        return t_alpha * tf.exp(- dist_xz)

    def diag_func(t_X):
        t_N = tf.shape(t_X)[0]
        return t_alpha * tf.ones([t_N], dtype=dtype)

    return Kernel(descriptor='Matern 1/2',
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=None)


def create_gp(t_X, t_Y, t_beta, kernel,
              t_hyper_prior=tf.constant(0.0, dtype=dtype),
              t_extra_prior=None):
    t_N = tf.shape(t_Y)[0]
    t_D = tf.shape(t_Y)[1]
    t_Q = tf.shape(t_X)[1]

    assert (issubclass(type(kernel), Kernel))

    jitter = 1.0e-8

    t_K_xx = kernel.covar_matrix(t_X, t_X) + ((1.0 / t_beta) + jitter) * tf.eye(t_N, dtype=dtype)

    t_L_xx = tf.cholesky(t_K_xx, name='L_xx')

    t_log_det = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(t_L_xx)), name='log_det_K_xx')

    t_data_fit = 0.5 * tf.reduce_sum(tf.square(
        tf.matrix_triangular_solve(t_L_xx, t_Y, lower=True)))

    t_noise_prior = 0.5 * tf.square(tf.log(t_beta), name='noise_prior')

    half_log_two_pi = tf.constant(0.5 * np.log(2.0 * np.pi), name='half_log_two_pi', dtype=dtype)

    t_neg_log_likelihood = half_log_two_pi * dtype_convert(t_D) * dtype_convert(t_N) + \
                           0.5 * dtype_convert(t_D) * t_log_det + \
                           t_data_fit + t_hyper_prior + t_noise_prior

    if t_extra_prior != None:
        t_neg_log_likelihood += t_extra_prior

    def create_prediction(tf_input):
        t_Kinv_Y = tf.matrix_triangular_solve(t_L_xx,
                                              tf.matrix_triangular_solve(t_L_xx, t_Y, lower=True),
                                              lower=True, adjoint=True)
        t_K_x_X = kernel.covar_matrix(tf_input, t_X)
        t_y_mean = tf.matmul(t_K_x_X, t_Kinv_Y)

        t_K_x_x_diag = tf.diag_part(kernel.covar_matrix(tf_input, tf_input)) + \
                       (1.0 / t_beta) * tf.ones([tf.shape(tf_input)[0]], dtype=dtype)

        t_y_var = t_K_x_x_diag - tf.reduce_sum(tf.square(
            tf.matrix_triangular_solve(t_L_xx, tf.transpose(t_K_x_X), lower=True)),
            axis=0)
        t_y_var = t_y_var[:, tf.newaxis]

        return t_y_mean, t_y_var

    t_prediction_placeholder = tf.placeholder(dtype=dtype)
    t_prediction_mean, t_prediction_var = create_prediction(t_prediction_placeholder)

    class GP(object):
        def __init__(self, t_neg_log_likelihood):
            self.name = 'GP'
            self.t_neg_log_likelihood = t_neg_log_likelihood

        def create_prediction(self, tf_input):
            return create_prediction(tf_input)

        @property
        def t_prediction_placeholder(self): return t_prediction_placeholder

        @property
        def t_prediction_mean(self): return t_prediction_mean

        @property
        def t_prediction_var(self): return t_prediction_var

        @property
        def t_input(self): return t_X

        @property
        def t_output(self): return t_Y

        @property
        def kernel(self): return kernel

    gp = GP(t_neg_log_likelihood=t_neg_log_likelihood)

    gp.t_extra_prior = t_extra_prior

    return gp


def create_sparse_gp(t_X, t_Y, t_beta,
                     kernel,
                     t_inducing_points,
                     t_hyper_prior=tf.constant(0.0, dtype=dtype),
                     t_extra_prior=None):
    t_N = tf.shape(t_Y)[0]
    t_D = tf.shape(t_Y)[1]
    t_Q = tf.shape(t_X)[1]

    assert (issubclass(type(kernel), Kernel))

    t_M = tf.shape(t_inducing_points)[0]
    t_Z = t_inducing_points

    jitter = 1.0e-8

    t_K_uu = kernel.covar_matrix(t_Z, t_Z) + (jitter) * tf.eye(t_M, dtype=dtype)
    t_L_uu = tf.cholesky(t_K_uu, name='L_uu')

    t_K_ux = kernel.covar_matrix(t_Z, t_X) #+ (jitter) * tf.eye(t_N, dtype=dtype)

    t_K_xx_diag = kernel.covar_diag(t_X)

    # Omega = K_xu L_uu^-T [M x N]
    t_OmegaT = tf.matrix_triangular_solve(t_L_uu, t_K_ux, lower=True)

    # C = Omega^T Omega [M x M]
    t_C = tf.matmul(t_OmegaT, t_OmegaT, transpose_b=True)

    t_A = t_C + ((1.0 / t_beta) + jitter) * tf.eye(t_M, dtype=dtype)
    t_L_aa = tf.cholesky(t_A, name='L_aa')

    # Gamma = Y^T Omega
    t_GammaT = tf.matmul(t_OmegaT, t_Y)

    # Psi = Gamma L_aa^-1
    t_PsiT = tf.matrix_triangular_solve(t_L_aa, t_GammaT, lower=True)

    t_yTy = tf.reduce_sum(tf.square(t_Y))

    half_log_two_pi = tf.constant(0.5 * np.log(2.0 * np.pi), name='half_log_two_pi', dtype=dtype)

    log_det_A = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(t_L_aa)))

    t_noise_prior = 0.5 * tf.square(tf.log(t_beta), name='noise_prior')

    t_neg_lower_bound = half_log_two_pi * dtype_convert(t_D) * dtype_convert(t_N) \
                        - 0.5 * dtype_convert(t_D) * dtype_convert(t_N - t_M) * tf.log(t_beta) \
                        + 0.5 * dtype_convert(t_D) * log_det_A \
                        + 0.5 * t_beta * t_yTy \
                        - 0.5 * t_beta * tf.reduce_sum(tf.square(t_PsiT)) \
                        + 0.5 * t_beta * (tf.reduce_sum(t_K_xx_diag) - tf.trace(t_C)) \
                        + t_hyper_prior \
                        + t_noise_prior

    if t_extra_prior != None:
        t_neg_lower_bound += t_extra_prior

    def create_prediction(tf_input):

        t_AInv = tf.cholesky_solve(t_L_aa, tf.eye(t_M, dtype=dtype))

        t_PredictAlpha = (1.0/t_beta) * tf.matrix_triangular_solve(t_L_uu,
                                                            tf.matmul(t_AInv, t_GammaT),
                                                            lower=True, adjoint=True)

        t_K_x_Z = kernel.covar_matrix(tf_input, t_Z)

        t_y_mean = t_beta * tf.matmul(t_K_x_Z, t_PredictAlpha)

        t_K_x_x_diag = kernel.covar_diag(tf_input)

        t_L_uuInv_K_Z_x = tf.matrix_triangular_solve(t_L_uu, tf.transpose(t_K_x_Z), lower=True)

        t_G = (1.0/t_beta) * t_AInv - tf.eye(t_M, dtype=dtype)

        t_y_var = t_K_x_x_diag \
                  + tf.reduce_sum(t_L_uuInv_K_Z_x * tf.matmul(t_G, t_L_uuInv_K_Z_x), axis=0) \
                  + (1.0 / t_beta) * tf.ones([tf.shape(tf_input)[0]], dtype=dtype)

        t_y_var = t_y_var[:, tf.newaxis]

        return t_y_mean, t_y_var

    t_prediction_placeholder = tf.placeholder(dtype=dtype)
    t_prediction_mean, t_prediction_var = create_prediction(t_prediction_placeholder)

    class GP(object):
        def __init__(self, t_neg_log_likelihood):
            self.name = 'SparseGP'
            self.t_neg_log_likelihood = t_neg_log_likelihood

        def create_prediction(self, tf_input):
            return create_prediction(tf_input)

        @property
        def t_prediction_placeholder(self): return t_prediction_placeholder

        @property
        def t_prediction_mean(self): return t_prediction_mean

        @property
        def t_prediction_var(self): return t_prediction_var

        @property
        def t_input(self): return t_X

        @property
        def t_output(self): return t_Y

    gp = GP(t_neg_log_likelihood=t_neg_lower_bound)

    gp.t_extra_prior = t_extra_prior

    gp.t_Z = t_Z

    return gp


def get_scaled_limits(X, scale_factor=1.1, axis=0):
    Xmin = np.min(X, axis=axis)
    Xmax = np.max(X, axis=axis)

    Q = len(Xmin)

    Xdim = Xmax - Xmin

    nXdim = np.max(Xdim) * np.ones([Q])

    Xmid = 0.5 * (Xmin + Xmax)

    Xmin = Xmid - scale_factor * 0.5 * nXdim
    Xmax = Xmid + scale_factor * 0.5 * nXdim

    return Xmin, Xmax

# Helper function for GP creation..
def create_standard_gp(t_X, t_Y, t_extra_prior=None):
    t_alpha = positive_variable(1.0)
    t_gamma = positive_variable(1.0)
    t_beta = positive_variable(1.0)
    kernel = create_squared_exp_kernel(t_alpha=t_alpha, t_gamma=t_gamma)
    t_hyper_prior = 0.5 * (tf.square(tf.log(t_alpha)) + tf.square(tf.log(t_gamma)))

    gp = create_gp(t_X, t_Y, t_beta, kernel=kernel, t_hyper_prior=t_hyper_prior,
                   t_extra_prior=t_extra_prior)

    gp.hyperparams = (t_alpha, t_gamma, t_beta)

    return gp


# Helper function for SparseGP creation..
def create_standard_sparse_gp(t_X, t_Y, num_inducing_points, num_input_dims, t_extra_prior=None):
    t_alpha = positive_variable(1.0)
    t_gamma = positive_variable(1.0)
    t_beta = positive_variable(1.0)
    kernel = create_squared_exp_kernel(t_alpha=t_alpha, t_gamma=t_gamma)
    t_hyper_prior = 0.5 * (tf.square(tf.log(t_alpha)) + tf.square(tf.log(t_gamma)))

    t_inducing_points = real_variable(init_linspace_inducing_points(num_inducing_points, num_input_dims))

    gp = create_sparse_gp(t_X, t_Y, t_beta,
                          kernel=kernel,
                          t_inducing_points=t_inducing_points,
                          t_hyper_prior=t_hyper_prior,
                          t_extra_prior=t_extra_prior)

    gp.hyperparams = (t_alpha, t_gamma, t_beta)

    return gp


def init_linspace_inducing_points(num_inducing_points, num_input_dims):
    return np.repeat(np.linspace(-1.0, 1.0, num_inducing_points)[:, np.newaxis],
                     num_input_dims, axis=1)


def test_align_full_gp_with_gp_or_energy(x_ref, y_ref,
                                         use_shared_beta=True,
                                         weights_prior_coeff=1.0,
                                         use_extra_gp_prior_over_warp=True,
                                         use_energy_version=False,
                                         energy_cost_weight=100.0,
                                         learning_rate=0.02,
                                         num_iterations=1200,
                                         use_zero_mean_Y_aligned=True,
                                         N_aligned_samples=None,
                                         use_sparse_gp=False,
                                         num_inducing_points=None, # Defaults to N_observed_samples
                                         use_shared_gamma=False,
                                         session=None,
                                         kernel_for_function='sq_exp'): # can also be 'matern12', 'matern32'
    if session == None:
        tf.reset_default_graph()

    if use_sparse_gp:
        assert (num_inducing_points != None)

    N_seq, N_samples = np.shape(x_ref)

    N_aligned_samples = N_aligned_samples or N_samples

    # The fixed, evenly spaced N_aligned_samples sample locations..
    x_fixed_samples = np.linspace(-1.0, 1.0, N_aligned_samples)[:, np.newaxis]

    # The fixed sample locations in tensorflow..
    t_X_fixed = tf.placeholder(dtype=dtype)
    feed_dict = {t_X_fixed: x_fixed_samples}
    

    t = np.linspace(-1.0, 1.0, N_samples)[:, np.newaxis]
    if use_extra_gp_prior_over_warp:
        # Uniformly spaced time for the gp prior over the warp (size N_samples)
        t_warp_prior_X = tf.placeholder(dtype=dtype)
        
        feed_dict[t_warp_prior_X] = t

        
    # Keep track of the final objective..
    t_objective = tf.constant(0.0, dtype=dtype)

    gps = []
    aligned_Y_list = []

    if use_energy_version:
        # Keep track of the reference error for the energy version..
        t_energy_error = tf.constant(0.0, dtype=dtype)

        # The estimated aligned mean for the energy version..
        t_Y_mean = real_variable(np.zeros([N_aligned_samples, 1]))

    # Use a shared noise model or not..
    if use_shared_beta:
        t_beta = positive_variable(1.0)

    if use_shared_gamma:
        t_gamma = positive_variable(1.0)

    print('kernel_for_function = ', kernel_for_function)
    for n, (x, y) in enumerate(zip(x_ref, y_ref)):

        # Ensure data has two dims..
        if (x.ndim == 1):
            x = x[:, np.newaxis]
        if (y.ndim == 1):
            y = y[:, np.newaxis]

        num_input_dims = x.shape[1]
        num_output_dims = y.shape[1]

        # Variables constrained to ensure a monotonic warp..
        t_X_unsmooth, t_X_raw = monotonic_sequence_variable(N_samples,
                                                            init=np.ones([N_samples, 1]))

        # Add extra GP prior over warps to be smooth..
        if use_extra_gp_prior_over_warp:
            if use_sparse_gp:
                gp_warp_prior = create_standard_sparse_gp(t_warp_prior_X, t_X_unsmooth,
                                                          num_inducing_points=num_inducing_points,
                                                          num_input_dims=1)
            else:
                gp_warp_prior = create_standard_gp(t_warp_prior_X, t_X_unsmooth)

            t_X, _ = gp_warp_prior.create_prediction(t_warp_prior_X)
        else:
            t_X = t_X_unsmooth

        # The unaligned reference signal to match..
        t_Y = tf.placeholder(dtype=dtype)

        # The variables estimating the aligned signal..
        t_Y_aligned = real_variable(np.zeros([N_aligned_samples, num_output_dims]))
        aligned_Y_list.append(t_Y_aligned)

        if use_energy_version:
            # Update the energe error with the squared error..
            t_energy_error += tf.reduce_sum(tf.square(t_Y_aligned - t_Y_mean))

        # Parameters for the GP..
        t_alpha = positive_variable(1.0)
        if not use_shared_gamma:
            t_gamma = positive_variable(1.0)

        # Option to use independent noise models..
        if not use_shared_beta:
            t_beta = positive_variable(1.0)

        if kernel_for_function == 'sq_exp':
            kernel = create_squared_exp_kernel(t_alpha=t_alpha, t_gamma=t_gamma)
            t_hyper_prior = 0.5 * (tf.square(tf.log(t_alpha)) + tf.square(tf.log(t_gamma)))

        elif kernel_for_function == 'matern12':
            kernel = create_matern12_kernel(t_alpha=t_alpha, t_gamma=t_gamma)
            t_hyper_prior = 0.5 * (tf.square(tf.log(t_alpha)) + tf.square(tf.log(t_gamma)))

        elif kernel_for_function == 'matern32':
            kernel = create_matern32_kernel(t_alpha=t_alpha, t_gamma=t_gamma)
            t_hyper_prior = 0.5 * (tf.square(tf.log(t_alpha)) + tf.square(tf.log(t_gamma)))
            
        elif kernel_for_function == 'periodic':
            t_period = positive_variable(10.)
            kernel = create_periodic_kernel(t_alpha=t_alpha, t_gamma=t_gamma, t_period=t_period)
            t_hyper_prior = 0.5 * (tf.square(tf.log(t_alpha)) + tf.square(tf.log(t_gamma)) + (tf.log(t_period)))
        
        elif kernel_for_function == 'matern12+periodic':
            t_alpha_1 = positive_variable(1.0)
            t_gamma_1 = positive_variable(1.0)
            t_beta = positive_variable(1.0)

            kernel_1 = create_matern12_kernel(t_alpha=t_alpha_1, t_gamma=t_gamma_1)

            t_alpha_4 = positive_variable(1.0)
            t_gamma_4 = positive_variable(1.0)
            t_period = positive_variable(1.)

            kernel_4 = create_periodic_kernel(t_alpha=t_alpha_4, t_gamma=t_gamma_4, t_period=t_period)

            # kernel = create_sum_kernel([kernel_1, kernel_2, kernel_3, kernel_4])
            kernel = create_sum_kernel([kernel_1, kernel_4])

            t_hyper_prior = 0.5 * (tf.square(tf.log(t_alpha_1)) + \
                           tf.square(tf.log(t_gamma_1)) + \
                           tf.square(tf.log(t_alpha_4)) + \
                           tf.square(tf.log(t_gamma_4)) + \
                           tf.square(tf.log(t_period)))
            
        elif kernel_for_function == 'matern32+periodic':
            t_alpha_1 = positive_variable(1.0)
            t_gamma_1 = positive_variable(1.0)
            t_beta = positive_variable(1.0)

            kernel_1 = create_matern32_kernel(t_alpha=t_alpha_1, t_gamma=t_gamma_1)

            t_alpha_4 = positive_variable(1.0)
            t_gamma_4 = positive_variable(1.0)
            t_period = positive_variable(1.)

            kernel_4 = create_periodic_kernel(t_alpha=t_alpha_4, t_gamma=t_gamma_4, t_period=t_period)

            # kernel = create_sum_kernel([kernel_1, kernel_2, kernel_3, kernel_4])
            kernel = create_sum_kernel([kernel_1, kernel_4])

            t_hyper_prior = 0.5 * (tf.square(tf.log(t_alpha_1)) + \
                           tf.square(tf.log(t_gamma_1)) + \
                           tf.square(tf.log(t_alpha_4)) + \
                           tf.square(tf.log(t_gamma_4)) + \
                           tf.square(tf.log(t_period)))

        else:
            raise NotImplementedError()

        # Combine the known and unknown inputs and outputs for each GP..
        t_X_combined = tf.concat([t_X, t_X_fixed], axis=0)
        t_Y_combined = tf.concat([t_Y, t_Y_aligned], axis=0)

        if use_sparse_gp:
            # t_inducing_points = real_variable(-1.0 + 2.0 * np.random.rand(num_inducing_points, num_input_dims))
            t_inducing_points = real_variable(init_linspace_inducing_points(num_inducing_points, num_input_dims))

            gp = create_sparse_gp(t_X=t_X_combined, t_Y=t_Y_combined, t_beta=t_beta,
                                  kernel=kernel,
                                  t_inducing_points=t_inducing_points,
                                  t_hyper_prior=t_hyper_prior)
        else:
            gp = create_gp(t_X=t_X_combined, t_Y=t_Y_combined, t_beta=t_beta,
                           kernel=kernel, t_hyper_prior=t_hyper_prior)

        # The noise free predictions for the ref (unaligned) signal reconstruction..
        t_predict_recon_mean, t_predict_recon_var = gp.create_prediction(t_X)

        # Place a prior over the raw warping weights to encourage smooth warps.
        # (This is a prior towards the identity warp as well)
        t_weights_prior = tf.reduce_sum(tf.square(t_X_raw))

        # Keep track of the feed dictionary for the reference signals (i.e. add the current one)..
        feed_dict.update({t_Y: y})

        # Add to the objective term..
        if use_extra_gp_prior_over_warp:
            t_objective += gp.t_neg_log_likelihood + gp_warp_prior.t_neg_log_likelihood
        else:
            t_objective += gp.t_neg_log_likelihood

        t_objective += weights_prior_coeff * t_weights_prior

        # Extra tensorflow vars in case we want to use for plotting things later..
        t_predict_input = tf.placeholder(dtype=dtype)
        t_predict_mean, t_predict_var = gp.create_prediction(t_predict_input)
        gp.predict = lambda x_test, feed_dict: predict(x_test, t_predict_input, feed_dict,
                                                       t_predict_mean, t_predict_var)

        # Keep track of extra variables in case we need to plot them..
        if use_extra_gp_prior_over_warp:
            gp.gp_warp_prior = gp_warp_prior
        gp.hyperparams = (t_alpha, t_gamma, t_beta)
        gp.t_X = t_X
        gp.t_X_raw = t_X_raw
        gp.t_Y_aligned = t_Y_aligned
        gp.t_X_combined = t_X_combined
        gp.t_Y_combined = t_Y_combined
        gp.t_predict_recon_mean = t_predict_recon_mean
        gp.t_predict_recon_var = t_predict_recon_var

        # Add to list of GPs..
        gps.append(gp)

    # Stack all the aligned estimates together into an (N_seq*num_output_dim) x N_sample matrix..
    t_Ys_aligned = tf.stack([tf.reshape(y, [-1]) for y in aligned_Y_list])
    t_Ys_aligned_mean = tf.constant(0.0, dtype=dtype)

    # Subtract off the mean..
    if use_zero_mean_Y_aligned:
        t_Ys_aligned_mean = tf.reduce_mean(t_Ys_aligned, axis=0, keep_dims=True)
        t_Ys_aligned = t_Ys_aligned - t_Ys_aligned_mean

    if use_energy_version:
        # Add the scaled energy cost to align the data (will implicitly set
        # t_Y_mean to the mean value since we are minimising the variance)..
        t_objective += energy_cost_weight * t_energy_error

        # Dummy replacement for compatibility on return values..
        alignment_gp = t_energy_error
    else:
        # Use a GPLVM with dim Q for the clustering of the aligned signals..
        Q = 2

        y_ref_multidim = y_ref.reshape([N_seq, N_samples * num_output_dims])

        X_aligned_init = standard_pca_initialisation(y_ref_multidim, Q)
        t_Xs_aligned = real_variable(X_aligned_init)

        # GPLVM std manifold prior..
        t_gplvm_prior = 0.01*tf.reduce_sum(tf.square(t_Xs_aligned))

        # The alignment GP that clusters the aligned estimates from the other GPs..
        alignment_gp = create_standard_gp(t_Xs_aligned, t_Ys_aligned,
                                          t_extra_prior=t_gplvm_prior)

        # Add the alignment GP to the objective..
        t_objective += alignment_gp.t_neg_log_likelihood

    def produce_plots(title, session, feed_dict):
        if use_energy_version:
            print('\nEnergy alignment squared error = ',
                  session.run(t_energy_error, feed_dict=feed_dict))
        else:
            alpha, gamma, beta = session.run(alignment_gp.hyperparams)
            print('\nAlignment GP',
                  ' alpha = %6.3f' % alpha,
                  ' gamma = %6.3f' % gamma,
                  ' beta = %6.3f (%0.3f)' % (beta, 1.0 / np.sqrt(beta)))

        max_per_row = 5
        num_rows = int(np.ceil(N_seq * 1.0 / max_per_row))
        num_cols = np.min([N_seq, max_per_row])
        plt.figure(figsize=[16, 4 * num_rows])
        plt.title(title)
        ax_to_share = None
        for n, (gp, c, x, y) in enumerate(zip(gps, sns.color_palette(n_colors=N_seq),
                                              x_ref, y_ref)):
            if (x.ndim == 1):
                x = x[:, np.newaxis]
            if (y.ndim == 1):
                y = y[:, np.newaxis]

            alpha, gamma, beta = session.run(gp.hyperparams)
            print('      GP %3d' % (n + 1),
                  ' alpha = %6.3f' % alpha,
                  ' gamma = %6.3f' % gamma,
                  ' beta = %8.3f (%0.3f)' % (beta, 1.0 / np.sqrt(beta)))

            x_opt = session.run(gp.t_X, feed_dict=feed_dict)

            ax = plt.subplot(num_rows, num_cols, n + 1, sharey=ax_to_share)
            ax_to_share = ax
            ax.plot(t, y, 'ko', label='obs data %d' % (n + 1))
            m, v = session.run((gp.t_predict_recon_mean, gp.t_predict_recon_var),
                               feed_dict=feed_dict)  # gp.predict(x_opt, feed_dict)
            ax.plot(t, m, '-', color=c, label='gp fit %d' % (n + 1))
            ax.plot(t, m + 2.0 * np.sqrt(v), ':', color=c)
            ax.plot(t, m - 2.0 * np.sqrt(v), ':', color=c)
            ax.legend()
            
            
            if use_sparse_gp:
                Z_pts = session.run(gp.t_Z, feed_dict=feed_dict)
                axis = ax.axis()
                if Z_pts.shape[1] == 1:
                    ax.plot(Z_pts[:,0], np.zeros(Z_pts[:,0].shape), 'k*')
                    ax.axis(axis)         
                    
     
        
        plt.figure(figsize=[16, 8])
        plt.title(title)
        ax = plt.subplot(2, 2, 1)
        for n, (gp, c, x, y) in enumerate(zip(gps, sns.color_palette(n_colors=N_seq),
                                              x_ref, y_ref)):
            x_opt = session.run(gp.t_X, feed_dict=feed_dict)
            # ax.plot(x, ':', color=c, label='true warp %d' % (n + 1))
            ax.plot(x_opt, '-', color=c)
        plt.title('Warps')

        ax = plt.subplot(2, 2, 2)
        for n, (gp, c, x, y) in enumerate(zip(gps, sns.color_palette(n_colors=N_seq),
                                              x_ref, y_ref)):
            y_unwarp = session.run(gp.t_Y_aligned, feed_dict=feed_dict)
            ax.plot(y_unwarp, '-', color=c, label='aligned %d' % (n + 1))
        plt.title('Aligned functions')
        
        ax = plt.subplot(2, 2, 3)
        for n, (gp, c, x, y) in enumerate(zip(gps, sns.color_palette(n_colors=N_seq),
                                              x_ref, y_ref)):
            x_raw = session.run(gp.t_X_raw)
            ax.plot(x_raw, '-', color=c, label='x_raw %d' % (n + 1))
        plt.title('X_raw')
        
        if not use_energy_version:

            def plot_manifold_heatmap(gp,
                                      session,
                                      feed_dict,
                                      fig=None):
                X = session.run(gp.t_input, feed_dict=feed_dict)
                Xmin, Xmax = get_scaled_limits(X, scale_factor=1.1, axis=0)

                varianceImageDim = 50
                ux, uy = np.meshgrid(np.linspace(Xmin[0], Xmax[0], varianceImageDim),
                                     np.linspace(Xmin[1], Xmax[1], varianceImageDim))

                xx = np.array([ux.flatten(), uy.flatten()]).transpose()
                feed_dict = feed_dict.copy()
                feed_dict[gp.t_prediction_placeholder] = xx

                var = session.run(gp.t_prediction_var, feed_dict=feed_dict)

                fig = fig or plt.figure()
                fig.imshow(var.reshape(varianceImageDim, varianceImageDim), interpolation='bicubic',
                           extent=(Xmin[0], Xmax[0], Xmin[1], Xmax[1]), aspect='auto', origin='lower',
                           cmap='RdBu')
                # plt.grid()
                # plt.title(title)
                # xlabel = xlabel or self.kern.to_string(session) + '\n' + self.noise_model.to_string(session)
                # plt.xlabel(xlabel)
                return fig

            X_aligned = session.run(t_Xs_aligned)
            ax = plt.subplot(2, 2, 4)
            plot_manifold_heatmap(alignment_gp, session, feed_dict, fig=plt)
            for n, (gp, c, x, y) in enumerate(zip(gps, sns.color_palette(n_colors=N_seq),
                                                  x_ref, y_ref)):
                ax.plot(X_aligned[n, 0], X_aligned[n, 1], 'o', color=c)
            plt.title('Manifold locations')

    print('use_energy_version = ', use_energy_version)
    if use_energy_version:
        print('energy_cost_weight = ', energy_cost_weight)
    print('use_shared_beta = ', use_shared_beta)
    print('use_shared_gamma = ', use_shared_gamma)
    print('weights_prior_coeff = ', weights_prior_coeff)
    print('use_extra_gp_prior_over_warp = ', use_extra_gp_prior_over_warp)
    print('learning_rate = ', learning_rate)
    print('num_iterations = ', num_iterations)
    print('use_zero_mean_Y_aligned = ', use_zero_mean_Y_aligned)
    print('N_aligned_samples = ', N_aligned_samples)
    print('use_sparse_gp = ', use_sparse_gp)
    if use_sparse_gp:
        print('num_inducing_points = ', num_inducing_points)

    # Run the optimisation!!
    print('\nRunning Optimisation!\n')

    def run_with_session(session):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(t_objective)

        session.run(tf.global_variables_initializer())

        start = time.time()
        refresh_iter = int(np.ceil(num_iterations / 10))
        for i in range(num_iterations):
            opt, cost = session.run((optimizer, t_objective),
                                    feed_dict=feed_dict)
            if (i % refresh_iter) == 0:
                print('  opt iter {:5}: {}'.format(i, cost))

        print('Final iter {:5}: {}'.format(i, cost))
        end = time.time()
        print('Time to optimise: {} s'.format(end - start))

        produce_plots(title='After Opt', session=session, feed_dict=feed_dict)

        return gps, alignment_gp, t_objective

    if session == None:
        with tf.Session() as session:
            return run_with_session(session)
    else:
        return run_with_session(session)


def perform_gplvm_unaligned_data(y_ref,
                                 learning_rate = 0.02,
                                 num_iterations = 1200,
                                 use_zero_mean_Y = True,
                                 num_latent_dimensions = 2):
    tf.reset_default_graph()

    N_seq, N_samples, num_output_dims = np.shape(y_ref)
    Q = num_latent_dimensions

    y_ref_multidim = y_ref.reshape([N_seq, N_samples * num_output_dims])
    print(y_ref_multidim.shape)

    X_init = standard_pca_initialisation(y_ref_multidim, Q)
    t_X = real_variable(X_init)

    t_Y = tf.placeholder(dtype=dtype)
    t_Y_mean = tf.constant(0.0, dtype=dtype)

    feed_dict = {t_Y: y_ref_multidim}

    # Subtract off the mean..
    if use_zero_mean_Y:
        t_Y_mean = tf.reduce_mean(t_Y, axis=0, keep_dims=True)
        t_Y = t_Y - t_Y_mean

    # GPLVM std manifold prior..
    t_gplvm_prior = tf.reduce_sum(tf.square(t_X))

    # The alignment GP that clusters the aligned estimates from the other GPs..
    alignment_gp = create_standard_gp(t_X, t_Y, t_extra_prior=t_gplvm_prior)

    # Add the alignment GP to the objective..
    t_objective = alignment_gp.t_neg_log_likelihood

    def produce_plots(title, session, feed_dict):
        def plot_manifold_heatmap(gp,
                                  session,
                                  feed_dict,
                                  fig=None):
            X = session.run(gp.t_input, feed_dict=feed_dict)
            Xmin, Xmax = get_scaled_limits(X, scale_factor=1.1, axis=0)

            varianceImageDim = 50
            ux, uy = np.meshgrid(np.linspace(Xmin[0], Xmax[0], varianceImageDim),
                                 np.linspace(Xmin[1], Xmax[1], varianceImageDim))

            xx = np.array([ux.flatten(), uy.flatten()]).transpose()
            feed_dict = feed_dict.copy()
            feed_dict[gp.t_prediction_placeholder] = xx

            var = session.run(gp.t_prediction_var, feed_dict=feed_dict)

            fig = fig or plt.figure()
            fig.imshow(var.reshape(varianceImageDim, varianceImageDim), interpolation='bicubic',
                       extent=(Xmin[0], Xmax[0], Xmin[1], Xmax[1]), aspect='auto', origin='lower',
                       cmap='RdBu')
            return fig

        plt.figure(figsize=[16, 4])
        plt.title(title)

        ax = plt.subplot(1, 2, 1)
        for n, (c, y) in enumerate(zip(sns.color_palette(n_colors=N_seq), y_ref)):
            ax.plot(y, '-', color=c, label='aligned %d' % (n + 1))
        plt.title('Unaligned functions')

        X_aligned = session.run(t_X)
        ax = plt.subplot(1, 2, 2)
        plot_manifold_heatmap(alignment_gp, session, feed_dict, fig=plt)
        for n, (c, y) in enumerate(zip(sns.color_palette(n_colors=N_seq), y_ref)):
            ax.plot(X_aligned[n, 0], X_aligned[n, 1], 'o', color=c)
        plt.title('Manifold locations')

    # Run the optimisation!!
    print('\nRunning Optimisation!\n')
    with tf.Session() as session:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(t_objective)

        session.run(tf.global_variables_initializer())

        start = time.time()
        refresh_iter = int(np.ceil(num_iterations / 10))
        for i in range(num_iterations):
            opt, cost = session.run((optimizer, t_objective),
                                    feed_dict=feed_dict)
            if (i % refresh_iter) == 0:
                print('  opt iter {:5}: {}'.format(i, cost))

        print('Final iter {:5}: {}'.format(i, cost))
        end = time.time()
        print('Time to optimise: {} s'.format(end - start))

        produce_plots(title='After Opt', session=session, feed_dict=feed_dict)

    return alignment_gp, t_objective


def test():
    D = 2  # = num output dimensions
    N_seq = 10
    N_samples = 50
    noise_std = 0.02

    np.random.seed(0)
    x_ref, y_ref, y_ref_noise_free = get_data(N_seq=N_seq,
                                              N_samples=N_samples,
                                              D=D,
                                              noise_std=noise_std)

    plt.figure(figsize=[18, 5])
    ax = plt.subplot(1, 2, 1)
    for x in x_ref:
        ax.plot(x)
    plt.title('Reference warps')

    def get_marker(d):
        markers = ['.', '+', ',', '-', '*', 'o']
        return markers[d % len(markers)]

    ax = plt.subplot(1, 2, 2)
    for yn, y, c in zip(y_ref, y_ref_noise_free, sns.color_palette(n_colors=N_seq)):
        for d in range(D):
            ax.plot(y[:, d], '-', color=c, marker=get_marker(d))
            ax.plot(yn[:, d], '+', color=c, marker=get_marker(d))
    plt.title('Unaligned signals (+ = noisy samples)')

    print('Input data of', N_seq, 'sequences of', N_samples, 'samples.')
    print(y.shape)

    print('N_seq = ', N_seq)
    print('N_samples = ', N_samples)
    print('noise_std = ', noise_std)
    print('num_output_dimensions = ', D)

    use_shared_beta = True
    weights_prior_coeff = 1.0
    use_extra_gp_prior_over_warp = True
    use_energy_version = False
    learning_rate = 0.01
    num_iterations = 1500
    use_zero_mean_Y_aligned = True
    N_aligned_samples = None
    use_sparse_gp = False

    gps, alignment_gp, t_objective = \
        test_align_full_gp_with_gp_or_energy(x_ref, y_ref,
                                             use_shared_beta=use_shared_beta,
                                             weights_prior_coeff=weights_prior_coeff,
                                             use_extra_gp_prior_over_warp=use_extra_gp_prior_over_warp,
                                             use_energy_version=use_energy_version,
                                             learning_rate=learning_rate,
                                             num_iterations=num_iterations,
                                             use_zero_mean_Y_aligned=use_zero_mean_Y_aligned,
                                             N_aligned_samples=N_aligned_samples,
                                             use_sparse_gp=use_sparse_gp)


if __name__ == '__main__':
    test()

    plt.show()