import sys, os
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

def diag_inverse(A):
    return tf.ones_like(A) / A

def einsum_trace(A):
    # trace -> sum of diagonal elements
    return tf.einsum('ij->i', A)

def diag_det(A):
    return tf.reduce_prod(A,axis=1)

def fast_symmetric_kl_div(X_mean, X_cov_diag, Y_mean, Y_cov_diag):
    def diag_inverse(A):
        return tf.ones_like(A) / A

    Y_cov_diag_inv = diag_inverse(Y_cov_diag)
    X_cov_diag_inv = diag_inverse(X_cov_diag)
   
    k = X_mean.get_shape().as_list()[1]
    
    trace_term_forward = tf.matmul(Y_cov_diag_inv, X_cov_diag, transpose_b=True)
    trace_term_backward = tf.transpose(tf.matmul(X_cov_diag_inv, Y_cov_diag, transpose_b=True))
    trace_term = trace_term_forward + trace_term_backward

    pairwise_mean_diff = tf.square(tf.expand_dims(Y_mean, 1) - tf.expand_dims(X_mean, 0))

    pairwise_cov_sum = tf.transpose(tf.expand_dims(X_cov_diag_inv, 1) + tf.expand_dims(Y_cov_diag_inv, 0),perm=[1,0,2])

    middle_term_einsum = tf.einsum('ijk,ijk->ij', pairwise_mean_diff, pairwise_cov_sum)
    #middle_term_forward = tf.einsum('ijk,ik->ij', pairwise_mean_diff,Y_cov_diag_inv)
    #middle_term_backward = tf.einsum('ijk,jk->ij', pairwise_mean_diff,X_cov_diag_inv)
    #middle_term = middle_term_forward + middle_term_backward
    kl_div = 0.5 * (trace_term + middle_term_einsum) - k
    return kl_div

def fast_kl_div(X_mean, X_cov_diag, Y_mean, Y_cov_diag):
    Y_cov_diag_inv = diag_inverse(Y_cov_diag)
    k = X_mean.get_shape().as_list()[1]
    trace_term = tf.matmul(Y_cov_diag_inv, X_cov_diag, transpose_b=True)
    pairwise_mean_diff = tf.transpose(tf.square(tf.expand_dims(Y_mean, 1) - tf.expand_dims(X_mean, 0)),perm=[1,0,2])
    

def main(argv):

    X_mean = np.array([[-2,-2,-2], [1,1,1]],dtype=np.float)
    Y_mean = np.array([[2,2,2], [0,0,0]],dtype=np.float)

    X_cov_diag = np.array([[1,1,1], [2,2,2]],dtype=np.float)
    Y_cov_diag = np.array([[3,3,3], [4,4,4]],dtype=np.float)
    
    X_mean_tf = tf.convert_to_tensor(X_mean)
    Y_mean_tf = tf.convert_to_tensor(Y_mean)
    
    X_cov_diag_tf = tf.convert_to_tensor(X_cov_diag)
    Y_cov_diag_tf = tf.convert_to_tensor(Y_cov_diag)

    Z_mean_tf = tf.zeros_like(X_mean)
    Z_cov_diag_tf = tf.ones_like(X_cov_diag)

    NX = tf.contrib.distributions.MultivariateNormalDiag(X_mean_tf, tf.sqrt(X_cov_diag_tf))
    NX0 = tf.contrib.distributions.MultivariateNormalDiag(X_mean_tf[0,:], tf.sqrt(X_cov_diag_tf[0,:]))
    NX1 = tf.contrib.distributions.MultivariateNormalDiag(X_mean_tf[1,:], tf.sqrt(X_cov_diag_tf[1,:]))
    NY = tf.contrib.distributions.MultivariateNormalDiag(Y_mean_tf, tf.sqrt(Y_cov_diag_tf))
    NY0 = tf.contrib.distributions.MultivariateNormalDiag(Y_mean_tf[0,:], tf.sqrt(Y_cov_diag_tf[0,:]))
    NY1 = tf.contrib.distributions.MultivariateNormalDiag(Y_mean_tf[1,:], tf.sqrt(Y_cov_diag_tf[1,:]))

    NZ = tf.contrib.distributions.MultivariateNormalDiag(Z_mean_tf, Z_cov_diag_tf)

    X0_to_Y0 = (NX0.kl_divergence(NY0) + NY0.kl_divergence(NX0)).numpy()
    X0_to_Y1 = (NX0.kl_divergence(NY1) + NY1.kl_divergence(NX0)).numpy()
    X1_to_Y0 = (NX1.kl_divergence(NY0) + NY0.kl_divergence(NX1)).numpy()
    X1_to_Y1 = (NX1.kl_divergence(NY1) + NY1.kl_divergence(NX1)).numpy()


    symm_kl_div_custom = fast_symmetric_kl_div(X_mean_tf, X_cov_diag_tf, Y_mean_tf, Y_cov_diag_tf).numpy()
    symm_kl_div = (NX.kl_divergence(NY) + NY.kl_divergence(NX)).numpy()

    print(symm_kl_div_custom)
    print("")
    print(symm_kl_div)
    print("Done!")
    print((NX.kl_divergence(NZ) + NZ.kl_divergence(NX)).numpy(), (NY.kl_divergence(NZ) + NZ.kl_divergence(NY)).numpy())
    print()
    print(fast_symmetric_kl_div(X_mean_tf, X_cov_diag_tf, Z_mean_tf, Z_cov_diag_tf).numpy(), fast_symmetric_kl_div(Y_mean_tf, Y_cov_diag_tf, Z_mean_tf, Z_cov_diag_tf).numpy())
    print()
    print(fast_symmetric_kl_div(X_mean_tf, X_cov_diag_tf, Y_mean_tf, Y_cov_diag_tf).numpy(), fast_symmetric_kl_div(Y_mean_tf, Y_cov_diag_tf, X_mean_tf, X_cov_diag_tf).numpy())
    print()
    print(np.argmin(symm_kl_div), np.argmin(symm_kl_div_custom))

    print()


if __name__ == "__main__":
    main(sys.argv[1:])