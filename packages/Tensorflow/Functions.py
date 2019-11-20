import tensorflow as tf

def symmetric_kl_div(X:tuple, Y:tuple):
    """
    Symmetric form of Kullback-Leibler divergence as KL(X,Y) + KL(Y,X)

    Parameters
    ----------
    X: tuple (mean, stddev) for variable X with dim k. \n
    Y: tuple (mean, stddev) for variable Y with dim k.

    Returns
    -------
    sym_kl_div: vector of length k holding the dimension wise symmetric KL divergence between X and Y. \n
    """

    Nx = tf.distributions.Normal(X[0], X[1], name='N_x')
    Ny = tf.distributions.Normal(Y[0], Y[1], name='N_y')

    KL_xy = tf.distributions.kl_divergence(Nx, Ny, name='kl_xy')
    KL_yx = tf.distributions.kl_divergence(Ny, Nx, name='kl_yx')

    # Equivalent computation
    # Nx.kl_divergence(Ny) + Ny.kl_divergence(Nx)

    sym_kl_div = KL_xy + KL_yx
    return sym_kl_div

def multivariate_kl_div(X:tf.distributions.Normal, Y:tf.distributions.Normal):
    """
    See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence
    for multivariate form, should output the same as other kl div computation functions.

    """

    cov_y = Y.covariance(name='cov_y')
    cov_y_inv = tf.linalg.inv(cov_y,name='cov_y_inv')
    cov_x = X.covariance(name='cov_x')
    mean_x = X.mean(name='mean_x')
    mean_y = Y.mean(name='mean_y')
    k = tf.shape(cov_x)[1]
    trace_term = tf.linalg.trace(tf.linalg.matmul(cov_y_inv, cov_x))
    middle_term =  tf.matmul(tf.transpose(mean_y - mean_x), tf.matmul(cov_y_inv, (mean_y - mean_x)))
    determinant_term = tf.log(tf.linalg.det(cov_y) / tf.linalg.det(cov_x))
    
    multi_kl_div = 0.5 * (trace_term + middle_term - k + determinant_term)
    return multi_kl_div
    

def bhattacharyya_distance(X, Y):
    """
    See https://en.wikipedia.org/wiki/Bhattacharyya_distance

    Simplified formulas:        \n
    (1):    D = B + 1/2 ln (A)
    (2):    B = (1/8) * B1^T * B2 * B1 \n
    (2.1):  B1 = (mean_x - mean_y) \n
    (2.2):  B2 = ((sigma_x + sigma_y) / 2) ^ (-1) \n
    (3):    A = A1 / A2 \n
    (3.1):  A1 = det((sigma_x + sigma_y) / 2) \n
    (3.2):  A2 = (det(sigma_x) + det(sigma_y))^(1/2)

    Parameters
    ----------
    X: tf.contrib.distributions.MultiVariateNormal* distribution. \n
    Y: tf.contrib.distributions.MultiVariateNormal* distribution.

    Returns
    -------
    b_dist: [batch_size,] vector filled with float

    """
    B1 = tf.expand_dims(X.mean() - Y.mean(),axis=-1)
    B2 = tf.linalg.inv((X.covariance() + Y.covariance())* 0.5)
    B = 0.125 * tf.squeeze(tf.matmul(B1, tf.linalg.matmul(B2,B1), transpose_a=True))
    
    A2 = tf.sqrt(tf.linalg.det(X.covariance()) + tf.linalg.det(Y.covariance()))
    A1 = tf.linalg.det((X.covariance() + Y.covariance()) * 0.5)
    A = A1 / A2

    b_dist = B + 0.5 * tf.log(A)
    return b_dist


def multivariate_squared_hellinger_distance(X, Y):
    """
    See https://en.wikipedia.org/wiki/Hellinger_distance for information on hellinger distance.
    X and Y are assumend to be mean vector of length K and covariance matrix with size KxK of multivariate normal distributions.
    
    Simplified formulas:        \n
    (1):    H2(P,Q)=1-A*exp(B)  \n
    (2):    A = A1/A2           \n
    (2.1):  A1 = det(sigma_x)^(1/4) * det(sigma_y)^(1/4) \n
    (2.2):  A2 = det((sigma_x + sigma_y)/2)^(1/2) \n
    (3):    B = -(1/8) * B1^T * B2 * B1 \n
    (3.1):  B1 = (mean_x - mean_y) \n
    (3.2):  B2 = ((sigma_x + sigma_y) / 2) ^ (-1)

    Parameters
    ----------
    X: tf.contrib.distributions.MultiVariateNormal* distribution. \n
    Y: tf.contrib.distributions.MultiVariateNormal* distribution.
    Returns
    -------
    h_squared: [batch_size,] vector filled with float
    """

    A1 = tf.pow(tf.linalg.det(X.covariance()),0.25) * tf.pow(tf.linalg.det(Y.covariance()),0.25)
    A2 = tf.sqrt(tf.linalg.det((X.covariance() + Y.covariance())* 0.5))
    A = A1/A2
    B1 = tf.expand_dims(X.mean() - Y.mean(),axis=-1)
    B2 = tf.linalg.inv((X.covariance() + Y.covariance())* 0.5)
    B = -0.125 * tf.squeeze(tf.matmul(B1, tf.linalg.matmul(B2,B1), transpose_a=True))
    h_squared = 1.0 - A * tf.exp(B)
    return h_squared


