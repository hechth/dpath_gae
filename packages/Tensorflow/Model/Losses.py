import tensorflow as tf

def latent_loss(mean, log_sigma_sq):
    """
    Function to compute the latent loss in variational autoencoders.
    
    Parameters
    ----------
        mean: tf.Tensor holding mean value of the distribution
        log_sigma_sq: tf.Tensor holding ln(sigma^2) of the distribution

    Returns
    -------
        KL divergence between multivariate normal and distribution defined by mean and variance.
    """
    # Latent loss
    # KL(P|Q) = sum over x [P(x) * log(P(x) / Q(x))]
    # Case for multivariate with unit gaussian prior:
    # KL = 1/2 * sum over i [sigma_i^2 + mu_i^2 - ln(sigma_i^2) - 1]
    latent_loss = -0.5 * tf.reduce_sum(1 + log_sigma_sq - tf.square(mean) - tf.exp(log_sigma_sq), axis=1)
    latent_loss = tf.reduce_mean(latent_loss)    
    return latent_loss
    #return tf.reduce_mean(-0.5 * tf.reduce_sum(1 + variance - tf.square(mean) - tf.log(variance), axis=1))

def multivariate_latent_loss(mean, covariance):
    #X = tf.contrib.distributions.MultivariateNormalFullCovariance(mean, covariance)
    X = tf.contrib.distributions.MultivariateNormalTriL(loc=mean, scale_tril=covariance)
    k = mean.get_shape().as_list()[-1]

    prior_mean = tf.zeros_like(mean)
    diagonal = tf.linalg.tensor_diag(tf.ones([k]))
    expanded = tf.expand_dims(diagonal,0)
    prior_covariance = tf.tile(expanded,[tf.shape(mean)[0],1,1])
    prior = tf.contrib.distributions.MultivariateNormalFullCovariance(prior_mean, prior_covariance)
    kl_div = X.kl_divergence(prior)
    return kl_div

def deformation_smoothness_loss(flow):
    """
    Computes a deformation smoothness based loss as described here:
    https://link.springer.com/content/pdf/10.1007%2F978-3-642-33418-4_16.pdf
    """

    dx, dy = tf.image.image_gradients(flow)

    dx2, dxy = tf.image.image_gradients(dx)
    dyx, dy2 = tf.image.image_gradients(dy)

    integral = tf.square(dx2) + tf.square(dy2) + tf.square(dxy) + tf.square(dyx)
    loss = tf.reduce_sum(integral)
    return loss