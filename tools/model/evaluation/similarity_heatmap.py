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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


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
        
        patch = normalize(tf.expand_dims(tf.image.crop_to_bounding_box(image,args.offsets[1], args.offsets[0],args.patch_size, args.patch_size),0))
        patch_code = tf.contrib.graph_editor.graph_replace(sess.graph.get_tensor_by_name('imported/code:0') ,{ sess.graph.get_tensor_by_name('imported/patch:0'): patch })
        patch_cov, patch_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_covariance_lower_tri/MatrixBandPart:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): patch })
        patch_distribution = tf.contrib.distributions.MultivariateNormalTriL(loc=patch_mean, scale_tril=patch_cov)
        
        # Size = (image_width - patch_size - 1) * (image_height - patch_size - 1) for 'VALID' padding and 
        # image_width * image_height for 'SAME' padding
        all_image_patches = tf.unstack(normalize(ctfi.extract_patches(image, args.patch_size, strides=[1,1,1,1],padding='SAME')))

        # Partition patches into chunks
        chunked_patches = list(chunks(all_image_patches, args.image_size[1]))
        chunked_patches = tf.stack(list(map(tf.stack, chunked_patches)))
        

        def compute_similarity_for_chunk(chunk_tensor):
            image_patches_cov, image_patches_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_covariance_lower_tri/MatrixBandPart:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): chunk_tensor })
            image_patches_distributions = tf.contrib.distributions.MultivariateNormalTriL(loc=image_patches_mean, scale_tril=image_patches_cov)
            similarities = patch_distribution.kl_divergence(image_patches_distributions) + image_patches_distributions.kl_divergence(patch_distribution)
            return similarities

        similarity_list = tf.map_fn(compute_similarity_for_chunk,chunked_patches, parallel_iterations=8)

        #for chunk in chunked_patches:
        #    image_patches = tf.stack(chunk)
        #    image_patches_cov, image_patches_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_covariance_lower_tri/MatrixBandPart:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): image_patches })      
        #    image_patches_distributions = tf.contrib.distributions.MultivariateNormalTriL(loc=image_patches_mean, scale_tril=image_patches_cov)
        #    similarities = patch_distribution.kl_divergence(image_patches_distributions) + image_patches_distributions.kl_divergence(patch_distribution)
        #    similarity_list.append(similarities)
        
        saver.restore(sess, latest_checkpoint)

        sim_vals = sess.run(similarity_list)
        plt.imshow(sim_vals)
        plt.show()

        print(sim_vals)
        print("Done!")
                


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main, argv=sys.argv[1:])