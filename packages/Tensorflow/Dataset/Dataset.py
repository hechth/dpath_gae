import tensorflow as tf

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