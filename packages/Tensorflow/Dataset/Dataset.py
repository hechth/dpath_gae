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