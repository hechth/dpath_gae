import tensorflow as tf

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

    # Iterate over all data, create encoded example and serialize it to the tf.Dataset
    for sample in data:
      example = func_encode(sample)
      writer.write(example.SerializeToString())

    # Flush and close the writer.
    writer.flush()
    writer.close()