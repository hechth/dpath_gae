import tensorflow as tf

def latent_loss(mean, variance):
    """
    Function to compute the latent loss in variational autoencoders.
    
    Parameters
    ----------
        mean: tf.Tensor holding mean value of the distribution
        variance: tf.Tensor holding variance of the distribution

    Returns
    -------
    KL divergence between multivariate normal and distribution defined by mean and variance.
    """
    # Latent loss
    # KL(P|Q) = sum over x [P(x) * log(P(x) / Q(x))]
    # Case for multivariate with unit gaussian prior:
    # KL = 1/2 * sum over i [sigma_i^2 + mu_i^2 - ln(sigma_i^2) - 1]
    return tf.reduce_mean(-0.5 * tf.reduce_sum(1 + variance - tf.square(mean) - tf.log(variance), axis=1))