import sys, argparse, os, math
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
import tensorflow as tf

from tensorflow.contrib import predictor

import packages.Tensorflow as ctf
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

    filename_target = os.path.join(git_root,'data','images','HE_level_1_cropped_512x512.png')
    image_target = tf.expand_dims(ctfi.load(filename_target, width=512, height=512, channels=3),0)

    filename_moving = os.path.join(git_root,'data','images','CD3_level_1_cropped_512x512.png')
    image_moving = tf.expand_dims(ctfi.load(filename_moving, width=512, height=512, channels=3),0)

    step = tf.Variable(tf.zeros([], dtype=tf.float32))    

    X, Y = np.mgrid[0:512:8j, 0:512:8j]
    positions = np.transpose(np.vstack([X.ravel(), Y.ravel()]))
    positions = tf.expand_dims(tf.convert_to_tensor(positions, dtype=tf.float32),0)

    target_source_control_point_locations = tf.Variable(positions)
    moving_source_control_point_locations = tf.Variable(positions)
    dest_control_point_locations = tf.Variable(positions)

    warped_moving = tf.Variable(image_moving)
    warped_moving, flow_moving = tf.contrib.image.sparse_image_warp(
        warped_moving,
        moving_source_control_point_locations,
        dest_control_point_locations,
        name='sparse_image_warp_moving',
        interpolation_order=2,
        regularization_weight=0.001,
        #num_boundary_points=1
    )

    warped_target = tf.Variable(image_target)
    warped_target, flow_target = tf.contrib.image.sparse_image_warp(
        warped_target,
        target_source_control_point_locations,
        dest_control_point_locations,
        name='sparse_image_warp_target',
        interpolation_order=2,
        regularization_weight=0.001,
        #num_boundary_points=1
    )


    warped_target_patches = normalize(ctfi.extract_patches(warped_target[0], 32, strides=[1,32,32,1]))
    warped_moving_patches = normalize(ctfi.extract_patches(warped_moving[0], 32, strides=[1,32,32,1]))

    #warped_target_patches = normalize(tf.image.extract_glimpse(tf.tile(warped_target,[64,1,1,1]),[32,32],target_source_control_point_locations[0]))
    #warped_moving_patches = normalize(tf.image.extract_glimpse(tf.tile(warped_moving,[64,1,1,1]),[32,32],moving_source_control_point_locations[0]))

    learning_rate = 0.0005

    with tf.Session(graph=tf.get_default_graph()).as_default() as sess:

        g = tf.Graph()       
        saved_model = predictor.from_saved_model(args.export_dir, graph=g)

        #fetch_ops = ['max_pooling2d_4/MaxPool:0','init']
        #fetch_ops = ['z:0','init']
        fetch_ops = ['z_mean/BiasAdd:0','z_log_sigma_sq/BiasAdd:0','init']
        fetch_ops.extend([v.name.strip(":0") + "/Assign" for v in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
                
        warped_target_graph = tf.graph_util.import_graph_def(g.as_graph_def(), input_map={'patch:0': warped_target_patches}, return_elements=fetch_ops, name='')
        warped_moving_graph = tf.graph_util.import_graph_def(g.as_graph_def(),input_map={'patch:0': warped_moving_patches}, return_elements=fetch_ops, name='')
        
        sess.run(warped_target_graph[2:])
        sess.run(warped_moving_graph[2:])

        #warped_target_codes = warped_target_graph[0]
        #warped_moving_codes = warped_moving_graph[0]

        target_mean = tf.squeeze(warped_target_graph[0])
        target_stddev = tf.sqrt(tf.exp(tf.squeeze(warped_target_graph[1])))
        target_distribution = (target_mean, target_stddev)
        N_target = tf.distributions.Normal(target_mean, target_stddev)

        moving_mean = tf.squeeze(warped_moving_graph[0])
        moving_stddev = tf.sqrt(tf.exp(tf.squeeze(warped_moving_graph[1])))
        moving_distribution = (moving_mean, moving_stddev)
        N_mov = tf.distributions.Normal(moving_mean, moving_stddev)

        sym_kl_div = ctf.symmetric_kl_div(target_distribution, moving_distribution)
        #multi_kl_div = ctf.multivariate_kl_div(N_target, N_mov)

        loss = tf.reduce_sum(sym_kl_div)
        

        #loss = tf.reduce_sum(tf.math.squared_difference(warped_target_codes, warped_moving_codes))
        #loss = tf.reduce_sum(tf.sqrt(tf.math.squared_difference(image_code, warped_code)))
        #loss = tf.reduce_sum(tf.math.squared_difference(warped_target, warped_moving))

        optimizer =  tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

        compute_gradients = optimizer.compute_gradients(loss,var_list=[moving_source_control_point_locations, target_source_control_point_locations])
        apply_gradients = optimizer.apply_gradients(compute_gradients, global_step=step)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        fig, ax = plt.subplots(3,3)
        ax[0,0].imshow(ctfi.rescale(image_target.eval(session=sess)[0], 0.0, 1.0))
        ax[0,0].set_title('target')
        ax[0,0].set_autoscale_on(False)

        ax[0,1].imshow(ctfi.rescale((image_target + image_moving).eval(session=sess)[0], 0.0, 1.0))
        ax[0,1].set_title('overlayed')
        ax[0,1].set_autoscale_on(False)

        ax[0,2].imshow(ctfi.rescale(image_moving.eval(session=sess)[0], 0.0, 1.0))
        ax[0,2].set_title('moving')
        ax[0,2].set_autoscale_on(False)

        plot_warped_target = ax[1,0].imshow(ctfi.rescale(warped_target.eval(session=sess)[0], 0.0, 1.0))
        ax[1,0].set_title('warped_target')
        ax[1,0].set_autoscale_on(False)

        plot_overlayed = ax[1,1].imshow(ctfi.rescale((warped_target + warped_moving).eval(session=sess)[0], 0.0, 1.0))
        ax[1,1].set_title('warped_overlayed')
        ax[1,1].set_autoscale_on(False)

        plot_warped_moving = ax[1,2].imshow(ctfi.rescale(warped_moving.eval(session=sess)[0], 0.0, 1.0))
        ax[1,2].set_title('warped_moving')
        ax[1,2].set_autoscale_on(False)

        plot_diff_target =ax[2,0].imshow(ctfi.rescale(tf.abs(image_target - warped_target).eval(session=sess)[0], 0., 1.))
        ax[2,0].set_title('diff_target')
        ax[2,0].set_autoscale_on(False)

        plot_diff_overlayed = ax[2,1].imshow(ctfi.rescale(tf.abs(warped_target - warped_moving).eval(session=sess)[0], 0., 1.))
        ax[2,1].set_title('diff_overlayed')
        ax[2,1].set_autoscale_on(False)

        plot_diff_moving = ax[2,2].imshow(ctfi.rescale(tf.abs(image_moving - warped_moving).eval(session=sess)[0], 0., 1.))
        ax[2,2].set_title('diff_moving')
        ax[2,2].set_autoscale_on(False)        

    
        dest_points = dest_control_point_locations.eval(session=sess)[0]
        moving_source_points = moving_source_control_point_locations.eval(session=sess)[0]
        target_source_points =target_source_control_point_locations.eval(session=sess)[0]

        plot_scatter_moving, = ax[1,2].plot(moving_source_points[:,0], moving_source_points[:,1],'s',marker='x', ms=5, color='orange')
        plot_scatter_target, = ax[1,0].plot(target_source_points[:,0], target_source_points[:,1],'s',marker='x', ms=5, color='orange')

        plot_moving_grad = ax[1,2].quiver(
            moving_source_points[:,0], # X
            moving_source_points[:,1], # Y
            np.zeros_like(moving_source_points[:,0]),
            np.zeros_like(moving_source_points[:,0]),
            units='xy',angles='xy', scale_units='xy', scale=1)

        plot_target_grad = ax[1,0].quiver(
            target_source_points[:,0], # X
            target_source_points[:,1], # Y
            np.zeros_like(target_source_points[:,0]),
            np.zeros_like(target_source_points[:,0]),
            units='xy',angles='xy', scale_units='xy', scale=1)

        plt.ion()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()

        iterations = 100000
        print_iterations = 100
        accumulated_gradients = np.zeros_like(sess.run(compute_gradients))

        while step.value().eval(session=sess) < iterations:
            step_val = int(step.value().eval(session=sess))

            gradients = sess.run(compute_gradients)
            sess.run(apply_gradients)

            accumulated_gradients += gradients

            if step_val % print_iterations == 0 or step_val == iterations - 1 :
                loss_val = loss.eval(session=sess)                
                
                diff_moving = tf.abs(image_moving - warped_moving).eval(session=sess)
                diff_target = tf.abs(image_target - warped_target).eval(session=sess)
                diff = tf.abs(warped_target - warped_moving).eval(session=sess)

                #warped_code_eval = np.mean(warped_moving_codes.eval(session=sess))

                print("{0:d}\t{1:.4f}\t{2:.4f}\t{3:.4f}\t{4:.4f}".format(step_val, loss_val, np.sum(diff_moving), np.sum(diff_target), np.sum(diff)))

                plot_warped_target.set_data(ctfi.rescale(warped_target.eval(session=sess)[0], 0., 1.))
                plot_warped_moving.set_data(ctfi.rescale(warped_moving.eval(session=sess)[0], 0., 1.))                
                plot_overlayed.set_data(ctfi.rescale((warped_target + warped_moving).eval(session=sess)[0], 0., 1.))

                plot_diff_target.set_data(ctfi.rescale(diff_target[0], 0., 1.))
                plot_diff_moving.set_data(ctfi.rescale(diff_moving[0], 0., 1.))
                plot_diff_overlayed.set_data(ctfi.rescale(diff[0], 0., 1.))

                moving_gradients = np.squeeze(accumulated_gradients[0][0])
                moving_points = np.squeeze(gradients[0][1])

                target_gradients = np.squeeze(accumulated_gradients[1][0])
                target_points = np.squeeze(gradients[1][1])


                plot_scatter_moving.set_data(moving_points[:,0], moving_points[:,1])
                plot_scatter_target.set_data(target_points[:,0], target_points[:,1])


                plot_moving_grad.remove()
                plot_moving_grad = ax[1,2].quiver(
                    moving_points[:,0], # X
                    moving_points[:,1], # Y
                    moving_gradients[:,0],
                    moving_gradients[:,1],
                    moving_gradients,
                    units='xy',angles='xy', scale_units='xy', scale=print_iterations)

                plot_target_grad.remove()
                plot_target_grad = ax[1,0].quiver(
                    target_points[:,0], # X
                    target_points[:,1], # Y
                    target_gradients[:,0],
                    target_gradients[:,1],
                    target_gradients,
                    units='xy',angles='xy', scale_units='xy', scale=print_iterations)

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.show()

                accumulated_gradients.fill(0)

        print("Done!")
        plt.ioff()
        plt.show()       
    
    sys.exit(0)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main, argv=sys.argv[1:])