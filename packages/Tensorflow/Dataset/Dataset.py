import sys, os, json
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

from psutil import virtual_memory

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

def construct_decode_op(cfg_inputs):
    """
    Function that based on the feature descriptions in a config file constructs the decode operation
    which deserializes the samples into a dict mapping the keys to the features

    Parameters
    ----------
    cfg_inputs: dict describing input configuration with 'features' and 'labels'
    """

    cfg_features = cfg_inputs['features']
    cfg_labels = cfg_inputs['labels']

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
        
        desc[cfg_labels['key']] = tf.FixedLenFeature(cfg_labels['shape'], cfg_labels['dtype'])

        return tf.parse_single_example(example_proto, desc)
    return decode_example

def construct_unzip_op(cfg_inputs):
    """
    Function that based on the feature descriptions in a config file constructs the final mapping operation
    which transforms the data in the right format.

    Parameters
    ----------
    cfg_inputs: dict describing input configuration with 'features' and 'labels'

    Returns
    -------
    unzip_example: callable which returns a pair (features, label).
    """

    cfg_features = cfg_inputs['features']
    cfg_labels = cfg_inputs['labels']

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
        for entry in cfg_features:
            features[entry['key']] = example_proto[entry['key']]
        label = example_proto[cfg_labels['key']]

        return (features, label)

    return unzip_example

def construct_train_fn(config):
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
    cfg_dataset = config['datasets']

    cfg_train_ds = cutil.safe_get('training', cfg_dataset)

    # Create operations
    decode_op = construct_decode_op(config)
    unzip_op = construct_unzip_op(config)

    operations = []
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
        dataset = dataset.map(decode_op)

        element_size = 0
        for output_type, output_shape  in zip(dataset.output_types.values(), dataset.output_shapes.values()):
            element_size += output_shape.num_elements() * output_type.size

        # Shuffle the dataset
        buffer_size = int(virtual_memory().total / 2 / element_size)
        dataset = dataset.shuffle(buffer_size)

        # Apply possible preprocessing, batch and prefetch the dataset.
        dataset = dataset.apply(tf.data.experimental.map_and_batch(preprocess, cfg_train_ds['batch'], num_parallel_batches=os.cpu_count()))
        dataset = dataset.prefetch(buffer_size=1)
        return dataset.repeat()

    return train_fn