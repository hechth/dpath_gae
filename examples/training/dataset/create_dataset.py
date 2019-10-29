import sys, os, argparse
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import packages.Utility as cutil
import packages.Tensorflow as ctf
import packages.Tensorflow.Dataset as ctfd

import numpy as np

import tensorflow as tf
tf.enable_eager_execution()

def main(argv):

    x_train = np.linspace(-100,100,num=100000)
    y_train = [np.math.sin(x) for x in x_train]

    data = zip(x_train, y_train)
   
    # Encoding function
    def func_encode(sample):
        x,y = sample
        features = { 'val': ctf.float_feature([x]), 'label': ctf.float_feature([y]) }
        return tf.train.Example(features=tf.train.Features(feature=features))

    filename = os.path.join(git_root,'examples','training','custom_dataset','training_ds.tfrecords')
    ctfd.write(data, func_encode, filename)
    cutil.publish(filename)
    


if __name__ == "__main__":
    main(sys.argv[1:])