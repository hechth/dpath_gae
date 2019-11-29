import sys, os, json
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)


import packages.Utility as cutil
import packages.Tensorflow as ctf
import packages.Tensorflow.Model as ctfm
import packages.Tensorflow.Image as ctfi

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

rotation_layer_conf = {
    "name":"rotor",
    "input":"angle",
    "layers":[
        {
            "type":"rotation",
            "image":"image_tensor",
            "interpolation":"BILINEAR",
            "name":"rotation_op"
        }
    ],
    "output":"rotated_image"
}

def main(argv):
    with tf.Session(graph=tf.get_default_graph()).as_default() as sess:
        filename = os.path.join(git_root, 'data','images','encoder_input.png')
        image = tf.expand_dims(ctfi.load(filename, width=32, height=32, channels=3),0,name='image_tensor')
        angle = tf.convert_to_tensor(np.random.rand(1,1), dtype=tf.float32, name='angle_tensor')
        tensors = {'image_tensor':image, 'angle': angle}
        rotation_layer = ctfm.parse_component(tensors, rotation_layer_conf, tensors)

        rotated_image = rotation_layer[2](angle)

        plt.imshow(sess.run(rotated_image)[0])
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])