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

def main(argv):
    parser = argparse.ArgumentParser(description='Compute codes and reconstructions for image.')
    parser.add_argument('export_dir',type=str,help='Path to saved model.')
    parser.add_argument('mean', type=str, help='Path to npy file holding mean for normalization.')
    parser.add_argument('variance', type=str, help='Path to npy file holding variance for normalization.')
    parser.add_argument('filename', type=str,help='Image file for which to create heatmap.')
    parser.add_argument('image_size', type=int, nargs=2, help='Size of the input image, HW.')
    parser.add_argument('offsets', type=int, nargs=2, help='Position where to extract the patch.')
    parser.add_argument('patch_size', type=int, help='Size of image patch.')
    #parser.add_argument('method', type=str, help='Method to use to measure similarity, one of KLD, SKLD, BD, HD, SQHD.')

    args = parser.parse_args()

    mean = np.load(args.mean)
    variance = np.load(args.variance)
    stddev = [np.math.sqrt(x) for x in variance]

    heatmap_height = args.image_size[0] - (args.patch_size - 1)
    heatmap_width = args.image_size[1] - (args.patch_size - 1)

    height_divisors = get_divisors(heatmap_height)

    if len(height_divisors) > 1:
        num_rows = height_divisors[1]
    else:
        num_rows = 10
    
    chunk_size = heatmap_width * 15

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
        image = ctfi.load(args.filename,height=args.image_size[0], width=args.image_size[1])
        
        patch = normalize(tf.expand_dims(tf.image.crop_to_bounding_box(image,args.offsets[0], args.offsets[1], args.patch_size, args.patch_size),0))
        patch_code = tf.contrib.graph_editor.graph_replace(sess.graph.get_tensor_by_name('imported/code:0') ,{ sess.graph.get_tensor_by_name('imported/patch:0'): patch })
        patch_cov, patch_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_covariance_lower_tri/MatrixBandPart:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): patch })
        patch_distribution = tf.contrib.distributions.MultivariateNormalTriL(loc=patch_mean, scale_tril=patch_cov)
        
        # Size = (image_width - patch_size - 1) * (image_height - patch_size - 1) for 'VALID' padding and
        # image_width * image_height for 'SAME' padding
        all_image_patches = tf.unstack(normalize(ctfi.extract_patches(image, args.patch_size, strides=[1,1,1,1], padding='VALID')))

        # Partition patches into chunks
        chunked_patches = list(create_chunks(all_image_patches, chunk_size))
        chunked_patches = list(map(tf.stack, chunked_patches))

        last_chunk = chunked_patches.pop()
        last_chunk_size = last_chunk.get_shape().as_list()[0]
        padding_size = chunk_size - last_chunk_size

        last_chunk = tf.concat([last_chunk,tf.tile(tf.zeros_like(patch),[padding_size,1,1,1])],0)

        chunk_tensor = tf.placeholder(tf.float32,shape=[chunk_size, args.patch_size, args.patch_size, 3], name='chunk_tensor_placeholder')
        image_patches_cov, image_patches_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_covariance_lower_tri/MatrixBandPart:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): chunk_tensor })
        image_patches_distributions = tf.contrib.distributions.MultivariateNormalTriL(loc=image_patches_mean, scale_tril=image_patches_cov)
        
        similarities = patch_distribution.kl_divergence(image_patches_distributions)# + image_patches_distributions.kl_divergence(patch_distribution)
        
        sim_vals = []

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, latest_checkpoint)

        for chunk in chunked_patches:
            sim_vals.extend(sess.run(similarities, feed_dict={chunk_tensor: sess.run(chunk)}))

        sim_vals.extend(sess.run(similarities, feed_dict={chunk_tensor: sess.run(last_chunk)})[:last_chunk_size])

        sim_heatmap = np.reshape(sim_vals[:heatmap_width * (heatmap_height - 1)], [heatmap_height - 1, heatmap_width])
        sim_vals_normalized = 1.0 - ctfi.rescale(sim_heatmap,0.0, 1.0)

        fig, ax = plt.subplots(1,3)

        denormalized_patch = denormalize(sess.run(patch)[0])
        max_sim_val = np.max(sim_vals)
        max_idx = np.argmin(sim_heatmap)
        max_idx = [int(max_idx / heatmap_width),int(max_idx % heatmap_width)]
        print(max_idx)
        ax[0].imshow(denormalized_patch)
        ax[1].imshow(sess.run(image))
        ax[2].imshow(sim_vals_normalized, cmap='plasma')
        
        plt.show()
        sess.close()
    print("Done!")
                


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main, argv=sys.argv[1:])