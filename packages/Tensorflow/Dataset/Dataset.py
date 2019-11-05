import sys, os, json
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

from psutil import virtual_memory
from objsize import get_deep_size

import tensorflow as tf

import packages.Utility as cutil


def encode(feature: dict):
    """
    Function to encode a feature with given name and encoding function.
    Returns
    -------
    A tf.train.Example holding the data stored with key from feature using the passed value
    """
    return tf.train.Example(features=tf.train.Features(feature=feature))

def decode(sample, desc:dict):
    """
    Function which decodes a sample using the feature description specified by desc.
    Returns
    -------
    dict containing the parsed example using the passed description.
    """
    return tf.parse_single_example(sample, desc)

def write(data, func_encode, filename):
    """
    Function which writes data to TFRecord file with filename.

    Parameters
    ----------
    data: iterable with elements

    func_encode: callable which returns a tf.train.Example

    filename: string specifying the location where to store the file
    """
    # Create a writer with the specified output filename
    writer = tf.io.TFRecordWriter(filename)

    # Create encoded dataset if function is provided.
    if func_encode is not None:
        encoded_data = [func_encode(sample).SerializeToString() for sample in data]
    else:
        encoded_data = data

    # Iterate over all data and serialize it to the tf.Dataset
    for sample in encoded_data:       
        writer.write(sample)      

    # Flush and close the writer.
    writer.flush()
    writer.close()

def construct_decode_op(cfg_features):
    """
    Function that based on the feature descriptions in a config file constructs the decode operation
    which deserializes the samples into a dict mapping the keys to the features

    Parameters
    ----------
    cfg_features: dict describing the 'features'
    """

    def decode_example(example_proto):
        """
        Function to decode an example by parsing the config to create the feature description.

        Parameters
        ----------
        example_proto: a tf.train.Example

        Returns
        -------
        dict mapping the decoded features
        """   
        desc = {}
        for entry in cfg_features:
            desc[entry['key']] = tf.FixedLenFeature(entry['shape'], entry['dtype'])     

        return tf.parse_single_example(example_proto, desc)
    return decode_example

def construct_unzip_op():
    """
    Function that based on the feature descriptions in a config file constructs the final mapping operation
    which transforms the data in the right format.

    Returns
    -------
    unzip_example: callable which returns a pair (features, label).
    """

    def unzip_example(example_proto):
        """
        Function that transforms a sample of the from
        { 'f0': val0, 'f1': val1, ..., 'label': val_label }
        to
        ({ 'f0': val0, 'f1': val1, ..., 'fx': valx }, val_label)

        Parameters
        ----------
        example_proto: dict of form { 'f0': val0, 'f1': val1, ..., 'label': val_label }

        Returns
        -------
        Pair of form ({ 'f0': val0, 'f1': val1, ..., 'fx': valx }, val_label)
        """   
        features = {}
        for key, value in example_proto.items():
            # Check for key with value 'label' to find the dict entry mapping to the label
            if key == 'label':
                label = value
            else:
                features[key] = value

        return (features, label)

    return unzip_example

def construct_train_fn(config, operations=[]):
    """
    Function to construct the training function based on the config.

    Parameters
    ----------
    config: dict holding model configuration.

    Returns
    -------
    train_fn: callable which is passed to estimator.train function.
    This function prepares the dataset and returns it in a format which is suitable for the estimator API.
    """

    cfg_train_ds = cutil.safe_get('training', config)

    # Create decode operation
    decode_op = construct_decode_op(config['features'])

    # Create unzip operation
    unzip_op = construct_unzip_op()

    operations.insert(0, decode_op)
    if 'operations' in cfg_train_ds:
        for op in cfg_train_ds['operations']:
            operations.append(cutil.get_function(op['module'], op['name']))

    operations.append(unzip_op)
    preprocess = cutil.concatenate_functions(operations)
   
    def train_fn():
        """
        Function which is passed to .train(...) call of an estimator object.

        Returns
        -------
        dataset: tf.data.Dataset object with elements ({'f0': v0, ... 'fx': vx}, label).
        """
        #Load the dataset
        dataset = tf.data.TFRecordDataset(cfg_train_ds['filename'])


        # Apply possible preprocessing, batch and prefetch the dataset.
        dataset = dataset.map(preprocess, num_parallel_calls=os.cpu_count())

        sample = tf.data.experimental.get_single_element(dataset.take(1))
        element_size = get_deep_size(sample)        

        # Shuffle the dataset
        buffer_size = tf.constant(int((virtual_memory().total / 2) / element_size), tf.int64)
        dataset = dataset.shuffle(buffer_size)

        dataset = dataset.batch(config['batch'])
        dataset = dataset.prefetch(buffer_size=1)
        return dataset.repeat()

    return train_fn

def estimate_mean_and_variance(dataset, num_samples, axes, feature) -> tuple:
    """
    Function to estimate the mean and variance of a feature using certain axes.

    Parameters
    ----------
        dataset: tf.data.TFRecordDataset object
        num_samples: int number of samples to use for estimation.
        axes: array holding the axes to use for estimation.
        feature: key identifying for which feature to estimate the normalization.

    Returns
    -------
        tf.nn.moments(samples[feature], axes=axes)
    """
    dataset = dataset.shuffle(num_samples)
    samples = tf.data.experimental.get_single_element(dataset.batch(num_samples))
    
    return tf.nn.moments(samples[feature], axes=axes)