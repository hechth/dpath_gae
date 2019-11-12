import sys, argparse, os, math
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor

import packages.Tensorflow.Image as ctfi
import packages.Utility as cutil

import matplotlib.pyplot as plt

mean = np.load("mean.npy")
variance = np.load("variance.npy")
stddev = [np.math.sqrt(x) for x in variance]

def normalize(image):
    channels = [tf.expand_dims((image[:,:,:,channel] - mean[channel]) / stddev[channel],-1) for channel in range(3)]
    return tf.concat(channels, 3)

def denormalize(image):
    channels = [np.expand_dims(image[:,:,channel] * stddev[channel] + mean[channel],-1) for channel in range(3)]
    denormalized_image = ctfi.rescale(np.concatenate(channels, 2), 0.0, 1.0)
    return denormalized_image


def main(argv):
    parser = argparse.ArgumentParser(description='Compute latent code for image patch by model inference.')
    parser.add_argument('export_dir',type=str,help='Path to saved model to use for inference.')

    args = parser.parse_args()

    filename = os.path.join(git_root,'data','images','tile_8_14.jpeg')
    image = tf.expand_dims(ctfi.load(filename, width=1024, height=1024, channels=3),0)

    true_angle = tf.Variable(initial_value=0.05 * math.pi, dtype=tf.float32, name='true_angle')  

    image_rotated = tf.Variable(image)
    image_rotated = tf.contrib.image.rotate(image_rotated, true_angle, interpolation='BILINEAR')

    step = tf.Variable(tf.zeros([], dtype=tf.float32))    

    X, Y = np.mgrid[0:1024:16j, 0:1024:16j]
    positions = np.transpose(np.vstack([X.ravel(), Y.ravel()]))
    positions = tf.expand_dims(tf.convert_to_tensor(positions, dtype=tf.float32),0)

    source_control_point_locations = tf.Variable(positions)
    dest_control_point_locations = tf.Variable(positions)

    warped_image = tf.Variable(image_rotated)
    warped_image, flow = tf.contrib.image.sparse_image_warp(
        warped_image,
        source_control_point_locations,
        dest_control_point_locations,
        name='sparse_image_warp',
        interpolation_order=2,
        regularization_weight=0.01
    )

    image_patches = normalize(ctfi.extract_patches(image[0], 32, strides=[1,32,32,1]))
    warped_patches = normalize(ctfi.extract_patches(warped_image[0], 32, strides=[1,32,32,1]))

    with tf.Session(graph=tf.get_default_graph()).as_default() as sess:
        g = tf.Graph()       
        saved_model = predictor.from_saved_model(args.export_dir, graph=g)

        fetch_ops = ['z:0','init']
        fetch_ops.extend([v.name.strip(":0") + "/Assign" for v in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
                
        image_graph = tf.graph_util.import_graph_def(g.as_graph_def(), input_map={'patch:0': image_patches}, return_elements=fetch_ops, name='')
        warped_graph = tf.graph_util.import_graph_def(g.as_graph_def(),input_map={'patch:0': warped_patches}, return_elements=fetch_ops, name='')
        
        sess.run(image_graph[1:])
        sess.run(warped_graph[1:])

        image_code = tf.constant(sess.run(image_graph[0]))

        loss = tf.reduce_sum(tf.math.squared_difference(image_code, warped_graph[0]))
        #loss = tf.reduce_sum(tf.math.squared_difference(image, warped_image))

        optimizer =  tf.train.GradientDescentOptimizer(learning_rate=0.001)
        compute_gradients = optimizer.compute_gradients(loss,var_list=[dest_control_point_locations])
        apply_gradients = optimizer.apply_gradients(compute_gradients, global_step=step)

        sess.run(tf.global_variables_initializer())


        fig, ax = plt.subplots(2,3)
        ax[0,0].imshow(ctfi.rescale(image.eval(session=sess)[0], 0.0, 1.0))
        ax[0,0].set_title('image')
        ax[0,1].imshow(ctfi.rescale(image_rotated.eval(session=sess)[0], 0.0, 1.0))
        ax[0,1].set_title('rotated')
        plot_warped = ax[0,2].imshow(ctfi.rescale(warped_image.eval(session=sess)[0], 0.0, 1.0))
        ax[0,2].set_title('warped')

        plot_diff_image =ax[1,0].imshow(ctfi.rescale(tf.abs(image - warped_image).eval(session=sess)[0], 0., 1.))
        ax[1,0].set_title('diff_image')
        plot_diff_rotated = ax[1,1].imshow(ctfi.rescale(tf.abs(image_rotated - warped_image).eval(session=sess)[0], 0., 1.))
        ax[1,1].set_title('diff_rotated')
        plot_flow = ax[1,2].imshow(np.zeros_like(image[0,:,:,:].eval(session=sess)))
        ax[1,2].set_title('flow')

        plt.ion()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()

        iterations = 2000
        while step.value().eval(session=sess) < iterations:
            gradients = sess.run(compute_gradients)
            step_val = int(step.value().eval(session=sess))
            sess.run(apply_gradients)
            if step_val % 100 == 0 or step_val == iterations - 1 :
                loss_val = loss.eval(session=sess)
                grad_mean = np.mean(gradients[0][0])
                flow_field = flow.eval(session=sess)
                x,y = np.split(ctfi.rescale(flow_field, 0.0, 1.0), 2, axis=3)
                flow_image = np.squeeze(np.stack([x,y,np.zeros_like(x)],axis=3))
                diff_warp_rotated = tf.abs(image_rotated - warped_image).eval(session=sess)
                diff_image_warp = tf.abs(image - warped_image).eval(session=sess)

                print("{0:d}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}\t{5:.4f}".format(step_val, loss_val, grad_mean, np.mean(flow_field), np.sum(diff_warp_rotated), np.sum(diff_image_warp)))

                plot_warped.set_data(ctfi.rescale(warped_image.eval(session=sess)[0], 0., 1.))
                plot_diff_image.set_data(ctfi.rescale(diff_image_warp[0], 0., 1.))
                plot_diff_rotated.set_data(ctfi.rescale(diff_warp_rotated[0], 0., 1.))
                plot_flow.set_data(flow_image)

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.show()

        print("Done!")
        plt.ioff()
        plt.show()       
    
    sys.exit(0)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main, argv=sys.argv[1:])