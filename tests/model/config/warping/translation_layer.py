import json
import os
import sys

import git

git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import packages.Tensorflow as ctf
import packages.Tensorflow.Image as ctfi
import packages.Tensorflow.Model as ctfm
import packages.Utility as cutil


translation_layer_conf = {
    "name":"translator",
    "input":"offset",
    "layers":[
        {
            "type":"translation",
            "image":"image_tensor",
            "interpolation":"BILINEAR",
            "name":"translation_op"
        }
    ],
    "output":"translated_image"
}

def main(argv):
    with tf.Session(graph=tf.get_default_graph()).as_default() as sess:
        filename = os.path.join(git_root, 'data','images','encoder_input.png')
        image = tf.expand_dims(ctfi.load(filename, width=32, height=32, channels=3),0,name='image_tensor')
        offset = tf.convert_to_tensor(np.random.rand(1,2), dtype=tf.float32, name='translation')
        tensors = {'image_tensor':image, 'offset': offset}
        translation_layer = ctfm.parse_component(tensors, translation_layer_conf, tensors)

        translated_image = translation_layer[2](offset)

        plt.imshow(sess.run(translated_image)[0])
        plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
