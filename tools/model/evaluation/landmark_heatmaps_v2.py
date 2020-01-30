import sys, argparse, os, math
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
from sklearn.metrics import mutual_info_score

import packages.Tensorflow as ctf
import packages.Tensorflow.Model as ctfm
import packages.Tensorflow.Image as ctfi
import packages.Utility as cutil

max_buffer_size_in_byte = 64*64*4*3*1000
max_patch_buffer_size = 2477273088

def _histogram_2d(a,b, bins):
  """
  takes two tensors of the same shape and computes the 2d histogram of their pairs
  """
  ar = a.reshape(-1)
  br = b.reshape(-1)
  aux = np.histogram2d(ar, br,bins=bins)
  return aux[0].astype(np.float32), aux[1].astype(np.float32), aux[2].astype(np.float32)

def get_patch_at(keypoint, images, patch_size):
    return tf.image.extract_glimpse(images, [patch_size, patch_size], [keypoint], normalized=False, centered=False)

def get_landmarks(filename, subsampling_factor=1):
    landmarks = []
    with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                landmarks.append([float(row[2]) / subsampling_factor, float(row[1]) / subsampling_factor])
            landmarks = np.array(landmarks,dtype=np.float32)
    return landmarks

def nmi_tf(x,y,bins):
    pxy, xedges, yedges = tf.py_func(_histogram_2d, [x, y, bins], [tf.float32, tf.float32, tf.float32])
    def _mi(mat):
        return mutual_info_score(None, None, contingency=mat).astype(np.float32)
    mi = tf.py_func(_mi, [pxy], tf.float32)
    return mi
    
def main(argv):
    parser = argparse.ArgumentParser(description='Compute similarity heatmaps of windows around landmarks.')
    parser.add_argument('export_dir',type=str,help='Path to saved model.')
    parser.add_argument('mean', type=str, help='Path to npy file holding mean for normalization.')
    parser.add_argument('variance', type=str, help='Path to npy file holding variance for normalization.')
    parser.add_argument('source_filename', type=str,help='Image file from which to extract patch.')
    parser.add_argument('source_image_size', type=int, nargs=2, help='Size of the input image, HW.')
    parser.add_argument('source_landmarks', type=str,help='CSV file from which to extract the landmarks for source image.')
    parser.add_argument('target_filename', type=str,help='Image file for which to create the heatmap.')
    parser.add_argument('target_image_size', type=int, nargs=2, help='Size of the input image for which to create heatmap, HW.')
    parser.add_argument('target_landmarks', type=str,help='CSV file from which to extract the landmarks for target image.')
    parser.add_argument('patch_size', type=int, help='Size of image patch.')
    parser.add_argument('output', type=str)
    parser.add_argument('--method', dest='method', type=str, help='Method to use to measure similarity, one of KLD, SKLD, BD, HD, SQHD.')
    parser.add_argument('--stain_code_size', type=int, dest='stain_code_size', default=0,
        help='Optional: Size of the stain code to use, which is skipped for similarity estimation')
    parser.add_argument('--rotate', type=float, dest='angle', default=0,
        help='Optional: rotation angle to rotate target image')
    parser.add_argument('--subsampling_factor', type=int, dest='subsampling_factor', default=1, help='Factor to subsample source and target image.')
    parser.add_argument('--region_size', type=int, default=64)
    args = parser.parse_args()

    mean = np.load(args.mean)
    variance = np.load(args.variance)
    stddev = [np.math.sqrt(x) for x in variance]

    def denormalize(image):
        channels = [np.expand_dims(image[:,:,channel] * stddev[channel] + mean[channel],-1) for channel in range(3)]
        denormalized_image = ctfi.rescale(np.concatenate(channels, 2), 0.0, 1.0)
        return denormalized_image

    def normalize(image, name=None, num_channels=3):
        channels = [tf.expand_dims((image[:,:,:,channel] - mean[channel]) / stddev[channel],-1) for channel in range(num_channels)]
        return tf.concat(channels, num_channels)

    latest_checkpoint = tf.train.latest_checkpoint(args.export_dir)   
    saver = tf.train.import_meta_graph(latest_checkpoint + '.meta', import_scope='imported')

    config = tf.ConfigProto()
    config.allow_soft_placement=True
    #config.log_device_placement=True

    # Load image and extract patch from it and create distribution.
    source_image = tf.expand_dims(ctfi.subsample(ctfi.load(args.source_filename,height=args.source_image_size[0], width=args.source_image_size[1]),args.subsampling_factor),0)
    args.source_image_size = list(map(lambda x: int(x / args.subsampling_factor), args.source_image_size))

    #Load image for which to create the heatmap
    target_image = tf.expand_dims(ctfi.subsample(ctfi.load(args.target_filename,height=args.target_image_size[0], width=args.target_image_size[1]),args.subsampling_factor),0)
    args.target_image_size = list(map(lambda x: int(x / args.subsampling_factor), args.target_image_size))

    source_landmarks = get_landmarks(args.source_landmarks, args.subsampling_factor)
    target_landmarks = get_landmarks(args.target_landmarks, args.subsampling_factor)

    region_size = args.region_size
    region_center = [int(region_size / 2),int(region_size / 2)]
    num_patches = region_size**2

    possible_splits = cutil.get_divisors(num_patches)
    num_splits = possible_splits.pop(0)

    while num_patches / num_splits > 512 and len(possible_splits) > 0:
        num_splits = possible_splits.pop(0)

    split_size = int(num_patches / num_splits)

    offset = 64
    center_idx = np.prod(region_center)

    X, Y = np.meshgrid(range(offset, region_size + offset), range(offset, region_size + offset))
    coords = np.concatenate([np.expand_dims(Y.flatten(),axis=1),np.expand_dims(X.flatten(),axis=1)],axis=1)

    coords_placeholder = tf.placeholder(tf.float32, shape=[split_size, 2])

    source_landmark_placeholder = tf.placeholder(tf.float32, shape=[1, 2])
    target_landmark_placeholder = tf.placeholder(tf.float32, shape=[1, 2])

    source_image_region = tf.image.extract_glimpse(source_image,[region_size + 2*offset, region_size+ 2*offset], source_landmark_placeholder, normalized=False, centered=False)
    target_image_region = tf.image.extract_glimpse(target_image,[region_size + 2*offset, region_size+ 2*offset], target_landmark_placeholder, normalized=False, centered=False)

    source_patches_placeholder = tf.map_fn(lambda x: get_patch_at(x, source_image, args.patch_size), source_landmark_placeholder, parallel_iterations=8, back_prop=False)[0]
    target_patches_placeholder = tf.squeeze(tf.map_fn(lambda x: get_patch_at(x, target_image_region, args.patch_size), coords_placeholder, parallel_iterations=8, back_prop=False))


    with tf.Session(config=config).as_default() as sess:
        saver.restore(sess, latest_checkpoint)

        source_patches_cov, source_patches_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_log_sigma_sq/BiasAdd:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): normalize(source_patches_placeholder) })
        source_patches_distribution = tf.contrib.distributions.MultivariateNormalDiag(source_patches_mean[:,args.stain_code_size:], tf.exp(source_patches_cov[:,args.stain_code_size:]))
        
        target_patches_cov, target_patches_mean = tf.contrib.graph_editor.graph_replace([sess.graph.get_tensor_by_name('imported/z_log_sigma_sq/BiasAdd:0'),sess.graph.get_tensor_by_name('imported/z_mean/BiasAdd:0')] ,{ sess.graph.get_tensor_by_name('imported/patch:0'): normalize(target_patches_placeholder) })
        target_patches_distribution = tf.contrib.distributions.MultivariateNormalDiag(target_patches_mean[:,args.stain_code_size:], tf.exp(target_patches_cov[:,args.stain_code_size:]))

        similarities_skld = source_patches_distribution.kl_divergence(target_patches_distribution) + target_patches_distribution.kl_divergence(source_patches_distribution)
        similarities_bd = ctf.bhattacharyya_distance(source_patches_distribution, target_patches_distribution)
        similarities_sad = tf.reduce_sum(tf.abs(source_patches_placeholder - target_patches_placeholder), axis=[1,2,3])

        source_patches_grayscale = tf.image.rgb_to_grayscale(source_patches_placeholder)
        target_patches_grayscale = tf.image.rgb_to_grayscale(target_patches_placeholder)

        similarities_nmi = tf.map_fn(lambda x: nmi_tf(tf.squeeze(source_patches_grayscale), tf.squeeze(x), 20), target_patches_grayscale)

        with open(args.output + "_" + str(region_size) + ".csv",'wt') as outfile:
            fp = csv.DictWriter(outfile, ["method", "landmark", "min_idx", "min_idx_value", "rank", "landmark_value"])
            methods = ["SKLD", "BD", "SAD", "MI"]
            fp.writeheader()
            
            results = []

            for k in range(len(source_landmarks)):

                heatmap_fused = np.ndarray((region_size, region_size, len(methods)))
                feed_dict={source_landmark_placeholder: [source_landmarks[k,:]], target_landmark_placeholder: [target_landmarks[k,:]] }
                
                for i in range(num_splits):
                    start = i * split_size
                    end = start + split_size
                    batch_coords = coords[start:end,:]

                    feed_dict.update({coords_placeholder: batch_coords})

                    similarity_values = np.array(sess.run([similarities_skld,similarities_bd, similarities_sad, similarities_nmi],feed_dict=feed_dict)).transpose()
                    #heatmap.extend(similarity_values)
                    for idx, val in zip(batch_coords, similarity_values):
                        heatmap_fused[idx[0] - offset, idx[1] - offset] = val

                for c in range(len(methods)):
                    heatmap = heatmap_fused[:,:,c]
                    if c == 3:
                        min_idx = np.unravel_index(np.argmax(heatmap),heatmap.shape)
                        min_indices = np.array(np.unravel_index(list(reversed(np.argsort(heatmap.flatten()))),heatmap.shape)).transpose().tolist()
                    else:
                        min_idx = np.unravel_index(np.argmin(heatmap),heatmap.shape)
                        min_indices = np.array(np.unravel_index(np.argsort(heatmap.flatten()),heatmap.shape)).transpose().tolist()

                    landmark_value = heatmap[region_center[0], region_center[1]]
                    rank = min_indices.index(region_center)

                    fp.writerow({"method": methods[c],"landmark": k, "min_idx": min_idx, "min_idx_value": heatmap[min_idx[0], min_idx[1]],"rank": rank , "landmark_value": landmark_value})
                    #matplotlib.image.imsave(args.output + "_" + str(region_size)+ "_"+ methods[c] + "_" + str(k) + ".jpeg", heatmap, cmap='plasma')
                outfile.flush()

                print(min_idx, rank)
        
        
            fp.writerows(results)


        sess.close()
        
    return 0

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main, argv=sys.argv[1:])