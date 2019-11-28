import sys, os, json
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import tensorflow as tf
import tensorflow.contrib.distributions as tfcd

import packages.Utility as cutil
import packages.Tensorflow as ctf
import packages.Tensorflow.Model as ctfm

import numpy as np


warping_layer_conf = {
    "name":"warper",
    "input":"texture",
    "layers":[
        {
            "type":"warping",
            "image": "texture",
            "flow":"deformation",
            "name":"dense_image_warp"
        }
    ],
    "output":"logits"
}

def main(argv):
    with tf.Session(graph=tf.get_default_graph()).as_default() as sess:
        image = tf.convert_to_tensor(np.random.rand(1,32,32,3), dtype=tf.float32, name='texture')
        flow = tf.convert_to_tensor(np.random.rand(1,32,32,2), dtype=tf.float32, name='deformation')
        tensors = {'texture':image, 'deformation': flow}
        warper = ctfm.parse_component(tensors, warping_layer_conf, tensors)

        warped_image = warper[2](image)

        print(sess.run(warped_image))


if __name__ == "__main__":
    main(sys.argv[1:])