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

def main(argv):
    parser = argparse.ArgumentParser(description='Compute latent code for image patch by model inference.')
    parser.add_argument('export_dir',type=str,help='Path to saved model to use for inference.')

    args = parser.parse_args()

    filename = os.path.join(git_root,'data','images','encoder_input.png')
    image = tf.expand_dims(ctfi.load(filename, width=32, height=32, channels=3),0)

    true_angle = tf.Variable(initial_value=0.05 * math.pi, dtype=tf.float32, name='true_angle')  

    image_rotated = tf.Variable(image)
    image_rotated = tf.contrib.image.rotate(image_rotated, true_angle, interpolation='BILINEAR')

    step = tf.Variable(tf.zeros([], dtype=tf.float32))    

    X, Y = np.mgrid[0:32:16j, 0:32:16j]
    positions = np.transpose(np.vstack([X.ravel(), Y.ravel()]))

    source_control_point_locations = tf.Variable(tf.expand_dims(tf.convert_to_tensor(positions, dtype=tf.float32),0))
    dest_control_point_locations = tf.Variable(tf.expand_dims(tf.convert_to_tensor(positions, dtype=tf.float32),0))

    warped_image = tf.Variable(image_rotated)
    warped_image, flow = tf.contrib.image.sparse_image_warp(
        warped_image,
        source_control_point_locations,
        dest_control_point_locations,
        name='sparse_image_warp'
    )

    

    with tf.Session(graph=tf.get_default_graph()).as_default() as sess:

        g = tf.Graph()       
        saved_model = predictor.from_saved_model(args.export_dir, graph=g)

        restore_ops = g.get_collection('saved_model_main_op')
        main_op = tf.saved_model.main_op_with_restore(restore_ops[0])


        #imported = tf.saved_model.load(
        #    sess,
        #    tags=[tf.saved_model.tag_constants.SERVING],
        #    export_dir=args.export_dir
        #)

        #base_op = tf.get_default_graph().get_operation_by_name('z')
        #print(sess.run(base_op.outputs[0], feed_dict={'patch:0': image.eval(session=sess)}))

        fetch_ops = ['z:0','init']
        fetch_ops.extend([v.name.strip(":0") + "/Assign" for v in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])

        image_graph = tf.graph_util.import_graph_def(g.as_graph_def(), input_map={'patch:0': image}, return_elements=fetch_ops, name='image')
        warped_graph = tf.graph_util.import_graph_def(g.as_graph_def(),input_map={'patch:0': warped_image}, return_elements=fetch_ops, name='warp')

        #image_init_op = tf.get_default_graph().get_operation_by_name('image/init')
        #warp_init_op = tf.get_default_graph().get_operation_by_name('warp/init')

        sess.run(image_graph[1])
        sess.run(warped_graph[1])



        loss = tf.reduce_sum(tf.math.squared_difference(image_graph[0], warped_graph[0]))

        optimizer =  tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        compute_gradients = optimizer.compute_gradients(loss,var_list=[dest_control_point_locations])
        apply_gradients = optimizer.apply_gradients(compute_gradients, global_step=step)

        sess.run(tf.global_variables_initializer())
        sess.run(image_graph[2:])
        sess.run(warped_graph[2:])

        while step.value().eval(session=sess) < 10000:
            sess.run(loss)
            gradients = sess.run(compute_gradients)
            if step.value().eval(session=sess) % 100 == 0:
                print(loss.eval(session=sess), np.mean(gradients[0][0]), np.mean(flow.eval(session=sess)), np.sum(tf.abs(image_rotated - warped_image).eval(session=sess)))
            sess.run(apply_gradients) 

        fig, ax = plt.subplots(2,2)
        ax[0,0].imshow(image.eval(session=sess)[0])
        ax[0,0].set_title('image')
        ax[0,1].imshow(image_rotated.eval(session=sess)[0])
        ax[0,1].set_title('rotated')
        plot_warp = ax[1,0].imshow(warped_image.eval(session=sess)[0])
        ax[1,0].set_title('warped')
        plot_diff = ax[1,1].imshow(ctfi.rescale(tf.abs(image_rotated - warped_image).eval(session=sess)[0], 0., 1.))
        ax[1,1].set_title('diff')
        plt.show()

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main, argv=sys.argv[1:])