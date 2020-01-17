import sys, argparse, os, math
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.contrib import predictor

import packages.Tensorflow as ctf
import packages.Tensorflow.Model as ctfm
import packages.Tensorflow.Image as ctfi
import packages.Utility as cutil

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
    kl_div = 0.5 * (trace_term + middle_term_einsum) - k
    return kl_div

def dist_kl(X,Y, structure_code_size):
    X_mean = X[:,0:structure_code_size]
    X_cov = tf.sqrt(tf.exp(X[:, structure_code_size:]))
    Y_mean = Y[:,0:structure_code_size]
    Y_cov = tf.sqrt(tf.exp(Y[:, structure_code_size:]))            
    return fast_symmetric_kl_div(X_mean, X_cov, Y_mean, Y_cov)

def create_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Method to print the divisors 
def get_divisors(n): 
    divisors = []       
    # List to store half of the divisors 
    for i in range(1, int(math.sqrt(n) + 1)) :          
        if (n % i == 0) :            
            divisors.append(int(n / i))
    divisors.append(1)
    return divisors

max_buffer_size_in_byte = 64*64*4*3*1000
max_patch_buffer_size = 2477273088

def main(argv):
    parser = argparse.ArgumentParser(description='Compute codes and reconstructions for image.')
    parser.add_argument('export_dir',type=str,help='Path to saved model.')
    parser.add_argument('mean', type=str, help='Path to npy file holding mean for normalization.')
    parser.add_argument('variance', type=str, help='Path to npy file holding variance for normalization.')
    parser.add_argument('source_filename', type=str,help='Image file from which to extract patch.')
    parser.add_argument('source_image_size', type=int, nargs=2, help='Size of the input image, HW.')
    parser.add_argument('offsets', type=int, nargs=2, help='Position where to extract the patch.')
    parser.add_argument('patch_size', type=int, help='Size of image patch.')
    parser.add_argument('target_filename', type=str,help='Image file for which to create the heatmap.')
    parser.add_argument('target_image_size', type=int, nargs=2, help='Size of the input image for which to create heatmap, HW.')
    parser.add_argument('method', type=str, help='Method to use to measure similarity, one of KLD, SKLD, BD, HD, SQHD.')
    parser.add_argument('--stain_code_size', type=int, dest='stain_code_size', default=0,
        help='Optional: Size of the stain code to use, which is skipped for similarity estimation')
    parser.add_argument('--rotate', type=float, dest='angle', default=0,
        help='Optional: rotation angle to rotate target image')
    parser.add_argument('--subsampling_factor', type=int, dest='subsampling_factor', default=1, help='Factor to subsample source and target image.')
    args = parser.parse_args()

    mean = np.load(args.mean)
    variance = np.load(args.variance)
    stddev = [np.math.sqrt(x) for x in variance]

    def denormalize(image):
        channels = [np.expand_dims(image[:,:,channel] * stddev[channel] + mean[channel],-1) for channel in range(3)]
        denormalized_image = ctfi.rescale(np.concatenate(channels, 2), 0.0, 1.0)
        return denormalized_image

    def normalize(image, name=None):
        channels = [tf.expand_dims((image[:,:,:,channel] - mean[channel]) / stddev[channel],-1) for channel in range(3)]
        return tf.concat(channels, 3, name=name)

    latest_checkpoint = tf.train.latest_checkpoint(args.export_dir)   
    saver = tf.train.import_meta_graph(latest_checkpoint + '.meta', import_scope='imported')

    with tf.Session(graph=tf.get_default_graph()).as_default() as sess:

        # Load image and extract patch from it and create distribution.
        source_image = ctfi.subsample(ctfi.load(args.source_filename,height=args.source_image_size[0], width=args.source_image_size[1]),args.subsampling_factor)
        args.source_image_size = list(map(lambda x: int(x / args.subsampling_factor), args.source_image_size))

        patch = normalize(tf.expand_dims(tf.image.crop_to_bounding_box(source_image,args.offsets[0], args.offsets[1], args.patch_size, args.patch_size),0))        
        
        patch_cov, patch_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_log_sigma_sq/BiasAdd:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): patch })
        #patch_distribution = tf.contrib.distributions.MultivariateNormalTriL(loc=patch_mean[:,args.stain_code_size:], scale_tril=patch_cov[:,args.stain_code_size:,args.stain_code_size:])
        patch_descriptor = tf.concat([patch_mean[:,args.stain_code_size:], tf.layers.flatten(patch_cov[:,args.stain_code_size:])], -1)
        sim_vals = []

        structure_code_size = patch_mean.get_shape().as_list()[1] - args.stain_code_size

        #Load image for which to create the heatmap
        target_image = ctfi.subsample(ctfi.load(args.target_filename,height=args.target_image_size[0], width=args.target_image_size[1]),args.subsampling_factor)
        args.target_image_size = list(map(lambda x: int(x / args.subsampling_factor), args.target_image_size))
        
        target_image = tf.contrib.image.rotate(target_image,np.radians(args.angle))

        heatmap_height = args.target_image_size[0] - (args.patch_size - 1)
        heatmap_width = args.target_image_size[1] - (args.patch_size - 1)
        
        # Compute byte size as: width*height*channels*sizeof(float32)
        patch_size_in_byte = args.patch_size**2 * 3 * 4
        max_patches = int(max_patch_buffer_size / patch_size_in_byte)
        max_num_rows = int(max_patches / heatmap_width)
        max_chunk_size = int(max_buffer_size_in_byte / patch_size_in_byte)

        #Iteration over image regions that we can load
        num_iterations = int(args.target_image_size[0] / max_num_rows) + 1
        
        all_chunks = list()
        all_similarities = list()
        chunk_tensors = list()

        chunk_sizes = np.zeros(num_iterations, dtype=np.int)
        chunk_sizes.fill(heatmap_width)

        for i in range(num_iterations):
            processed_rows = i * max_num_rows
            rows_to_load = min(max_num_rows + (args.patch_size - 1), args.target_image_size[0] - processed_rows)
            if rows_to_load < args.patch_size:
                break

            # Extract region for which we can compute patches        
            target_image_region = tf.image.crop_to_bounding_box(target_image, processed_rows, 0, rows_to_load, args.target_image_size[1])
    
            # Size = (image_width - patch_size - 1) * (image_height - patch_size - 1) for 'VALID' padding and
            # image_width * image_height for 'SAME' padding
            all_image_patches = tf.unstack(normalize(ctfi.extract_patches(target_image_region, args.patch_size, strides=[1,1,1,1], padding='VALID')))

            possible_chunk_sizes = get_divisors(len(all_image_patches))
            for size in possible_chunk_sizes:
                if size < max_chunk_size:
                    chunk_sizes[i] = size
                    break

            # Partition patches into chunks
            chunked_patches = list(create_chunks(all_image_patches, chunk_sizes[i]))
            chunked_patches = list(map(tf.stack, chunked_patches))
            all_chunks.append(chunked_patches)

            chunk_tensor = tf.placeholder(tf.float32,shape=[chunk_sizes[i], args.patch_size, args.patch_size, 3], name='chunk_tensor_placeholder')
            chunk_tensors.append(chunk_tensor)
            
            image_patches_cov, image_patches_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_log_sigma_sq/BiasAdd:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): chunk_tensor })
            image_patches_descriptors = tf.concat([image_patches_mean[:,args.stain_code_size:], tf.layers.flatten(image_patches_cov[:,args.stain_code_size:])], -1)
        
            similarities = tf.squeeze(dist_kl(patch_descriptor, image_patches_descriptors, structure_code_size))

            all_similarities.append(similarities)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, latest_checkpoint)

        for i in range(len(all_chunks)):
            for chunk in all_chunks[i]:
                #chunk_vals = sess.run(all_similarities[i], feed_dict={chunk_tensors[i]: sess.run(chunk)})
                sim_vals.extend(sess.run(all_similarities[i], feed_dict={chunk_tensors[i]: sess.run(chunk)}))

        print(len(sim_vals))
        sim_heatmap = np.reshape(sim_vals, [heatmap_height, heatmap_width])
        heatmap_tensor = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(sim_heatmap),-1),0)
        dy, dx = tf.image.image_gradients(heatmap_tensor)
        sim_vals_normalized = 1.0 - ctfi.rescale(sim_heatmap,0.0, 1.0)

        k_min = 20
        min_indices = np.unravel_index(np.argsort(sim_vals)[:k_min],sim_heatmap.shape)
        fig_min, ax_min = plt.subplots(4,5)

        for i in range(k_min):
            target_patch = tf.image.crop_to_bounding_box(target_image, int(min_indices[0][i]), int(min_indices[1][i]), args.patch_size, args.patch_size)
            ax_min[int(i / 5), int(i % 5)].imshow(sess.run(target_patch))
            ax_min[int(i / 5), int(i % 5)].set_title('y:' + str(min_indices[0][i]) + ', x:' + str(min_indices[1][i]))

        fig, ax = plt.subplots(2,3)
        cmap = 'plasma'

        denormalized_patch = denormalize(sess.run(patch)[0])
        max_sim_val = np.max(sim_vals)
        max_idx = np.unravel_index(np.argmin(sim_heatmap),sim_heatmap.shape)

        target_image_patch = tf.image.crop_to_bounding_box(target_image, max_idx[0], max_idx[1], args.patch_size, args.patch_size)
        print(max_idx)

        
        print(min_indices)
        ax[1,0].imshow(sess.run(source_image))
        ax[1,1].imshow(sess.run(target_image))
        ax[0,0].imshow(denormalized_patch)
        heatmap_image = ax[0,2].imshow(sim_heatmap, cmap=cmap)
        ax[0,1].imshow(sess.run(target_image_patch))
        #dx_image = ax[0,2].imshow(np.squeeze(sess.run(dx)), cmap='bwr')
        #dy_image = ax[1,2].imshow(np.squeeze(sess.run(dy)), cmap='bwr')
        gradient_image = ax[1,2].imshow(np.squeeze(sess.run(dx+dy)), cmap='bwr')

        fig.colorbar(heatmap_image, ax=ax[0,2])
        #fig.colorbar(dx_image, ax=ax[0,2])
        #fig.colorbar(dy_image, ax=ax[1,2])
        fig.colorbar(gradient_image, ax=ax[1,2])
        
        plt.show()
        sess.close()
    print("Done!")
                


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main, argv=sys.argv[1:])