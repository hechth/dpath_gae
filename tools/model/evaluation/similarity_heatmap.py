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
    return divisors

max_buffer_size_in_byte = 59105280
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

    args = parser.parse_args()

    mean = np.load(args.mean)
    variance = np.load(args.variance)
    stddev = [np.math.sqrt(x) for x in variance]

    heatmap_height = args.target_image_size[0] - (args.patch_size - 1)
    heatmap_width = args.target_image_size[1] - (args.patch_size - 1)

        
    # Compute byte size as: width*height*channels*sizeof(float32)
    patch_size_in_byte = args.patch_size**2 * 3 * 4
    max_patches = int(max_patch_buffer_size / patch_size_in_byte)
    max_num_rows = int(max_patches / heatmap_width)
    max_chunk_size = int(max_buffer_size_in_byte / patch_size_in_byte)


    chunk_size = heatmap_width

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
        source_image = ctfi.load(args.source_filename,height=args.source_image_size[0], width=args.source_image_size[1])
        patch = normalize(tf.expand_dims(tf.image.crop_to_bounding_box(source_image,args.offsets[0], args.offsets[1], args.patch_size, args.patch_size),0))        
        patch_cov, patch_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_covariance_lower_tri/MatrixBandPart:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): patch })
        patch_distribution = tf.contrib.distributions.MultivariateNormalTriL(loc=patch_mean, scale_tril=patch_cov)
        
        sim_vals = []

        #Load image for which to create the heatmap
        target_image = ctfi.load(args.target_filename,height=args.target_image_size[0], width=args.target_image_size[1])

        #Iteration over image regions that we can load
        num_iterations = int(args.target_image_size[0] / max_num_rows) + 1
        for i in range(num_iterations):
            processed_rows = i * max_num_rows
            rows_to_load = min(max_num_rows + (args.patch_size - 1), args.target_image_size[0] - processed_rows)

            # Extract region for which we can compute patches        
            target_image_region = tf.image.crop_to_bounding_box(target_image, processed_rows, 0, rows_to_load, args.target_image_size[1])
    
            # Size = (image_width - patch_size - 1) * (image_height - patch_size - 1) for 'VALID' padding and
            # image_width * image_height for 'SAME' padding
            all_image_patches = tf.unstack(normalize(ctfi.extract_patches(target_image_region, args.patch_size, strides=[1,1,1,1], padding='VALID')))

            # Partition patches into chunks
            chunked_patches = list(create_chunks(all_image_patches, chunk_size))
            chunked_patches = list(map(tf.stack, chunked_patches))

            #last_chunk = chunked_patches.pop()
            #last_chunk_size = last_chunk.get_shape().as_list()[0]
            #padding_size = chunk_size - last_chunk_size
            #last_chunk_padded = tf.concat([last_chunk, tf.ones([padding_size, args.patch_size, args.patch_size, 3],dtype=tf.float32)],0)

            chunk_tensor = tf.placeholder(tf.float32,shape=[chunk_size, args.patch_size, args.patch_size, 3], name='chunk_tensor_placeholder')
            image_patches_cov, image_patches_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_covariance_lower_tri/MatrixBandPart:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): chunk_tensor })
            image_patches_distributions = tf.contrib.distributions.MultivariateNormalTriL(loc=image_patches_mean, scale_tril=image_patches_cov)
        
            if args.method == 'SKLD':
                similarities = patch_distribution.kl_divergence(image_patches_distributions) + image_patches_distributions.kl_divergence(patch_distribution)
            elif args.method == 'BD':
                similarities = ctf.bhattacharyya_distance(patch_distribution, image_patches_distributions)
            elif args.method == 'SQHD':
                similarities = ctf.multivariate_squared_hellinger_distance(patch_distribution, image_patches_distributions)
            elif args.method == 'HD':
                similarities = tf.sqrt(ctf.multivariate_squared_hellinger_distance(patch_distribution, image_patches_distributions))
            else:
                similarities = patch_distribution.kl_divergence(image_patches_distributions)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, latest_checkpoint)

            for chunk in chunked_patches:
                chunk_vals = sess.run(similarities, feed_dict={chunk_tensor: sess.run(chunk)})
                sim_vals.extend(chunk_vals)
                #sim_vals.extend(np.random.rand(chunk_size))

            #last_chunk_eval = sess.run(last_chunk)
            #last_chunk_vals = sess.run(similarities, feed_dict={chunk_tensor: sess.run(last_chunk_padded)})
            #sim_vals.extend(np.zeros(last_chunk_size))
            #sim_vals.extend(last_chunk_vals[0:last_chunk_size])

        print(len(sim_vals))
        sim_heatmap = np.reshape(sim_vals, [heatmap_height, heatmap_width])
        sim_vals_normalized = 1.0 - ctfi.rescale(sim_heatmap,0.0, 1.0)

        fig, ax = plt.subplots(2,2)
        cmap = 'plasma'

        denormalized_patch = denormalize(sess.run(patch)[0])
        max_sim_val = np.max(sim_vals)
        max_idx = np.argmin(sim_heatmap)
        max_idx = [int(max_idx / heatmap_width),int(max_idx % heatmap_width)]
        print(max_idx)
        ax[1,0].imshow(sess.run(source_image))
        ax[1,1].imshow(sess.run(target_image))
        ax[0,0].imshow(denormalized_patch)
        heatmap_image = ax[0,1].imshow(sim_vals_normalized, cmap=cmap)

        fig.colorbar(heatmap_image, ax=ax[0,1])
        
        plt.show()
        sess.close()
    print("Done!")
                


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main, argv=sys.argv[1:])