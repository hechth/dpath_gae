import sys, os, argparse, re, itertools
import git
sys.path.append(git.Repo('.', search_parent_directories=True).working_tree_dir)

import tensorflow as tf
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()

import packages.Utility as cutil

import packages.Tensorflow as ctf
import packages.Tensorflow.Dataset as ctfd
import packages.Tensorflow.Image as ctfi


def _decode_example_filename(example_proto):
    image_feature_description = {     
        'filename': tf.FixedLenFeature([], tf.string),
    }
    return tf.parse_single_example(example_proto, image_feature_description)

   

def main(argv):
    parser = argparse.ArgumentParser(description='Create tfrecords dataset holding patches of images specified by filename in input dataset.')

    parser.add_argument('input_dataset', type=str, help='Path to dataset holding image filenames')
    parser.add_argument('output_dataset', type=str, help='Path where to store the output dataset')
    parser.add_argument('patch_size', type=int, help='Patch size which to use in the preprocessed dataset')
    parser.add_argument('num_samples', type=int, help='Size of output dataset')
    parser.add_argument('labels', type=lambda s: [item for item in s.split(',')], help="Comma separated list of labels to find in filenames.")
    parser.add_argument('--image_size', type=int, dest='image_size', help='Image size for files pointed to by filename')
    parser.add_argument('--threshold', type=float, dest='threshold', help='Threshold for filtering the samples according to variation.')

    args = parser.parse_args()

    labels_table = tf.contrib.lookup.index_table_from_tensor(mapping=args.labels)
    
    filename_dataset = tf.data.TFRecordDataset(args.input_dataset, num_parallel_reads=8).map(_decode_example_filename).shuffle(100000)

    functions = [tf.Variable(label, name='const_'+ label).value for label in args.labels]
    false_fn = tf.Variable('None', name='none_label').value
    
    
    def _extract_label(filename):
        #base_size = tf.size(tf.string_split([filename],""))
        #predicates = [tf.equal(base_size, tf.size(tf.string_split([tf.regex_replace(filename, "/"+ label + "/", "")])))  for label in args.labels]
        
        match = [tf.math.reduce_any(tf.strings.regex_full_match(tf.string_split([filename],'/').values,label)) for label in args.labels]        
        pred_fn_pairs = list(zip(match,functions))
        return tf.case(pred_fn_pairs, default=false_fn, exclusive=True)



    # Load images and extract the label from the filename
    if args.image_size is not None:
        images_dataset = filename_dataset.map(lambda feature: {'image': ctfi.load(feature['filename'], channels=3, width=args.image_size, height=args.image_size), 'label': labels_table.lookup(_extract_label(feature['filename']))})
    else:
        images_dataset = filename_dataset.map(lambda feature: {'image': ctfi.load(feature['filename'], channels=3), 'label': labels_table.lookup(_extract_label(feature['filename']))})

    def _filter_func_label(features):
        label = features['label']
        result = label > -1
        return result
        
    images_dataset = images_dataset.filter(_filter_func_label).shuffle(500)
    # Extract image patches

    #for sample in tfe.Iterator(images_dataset):
    #    print(sample['label'])

    def _split_patches(features):
        patches = ctfi.extract_patches(features['image'], args.patch_size)
        labels = tf.expand_dims(tf.reshape(features['label'], [1]),0)
        labels = tf.tile(labels,tf.stack([tf.shape(patches)[0], 1]))
        return (patches, labels)

    patches_dataset = images_dataset.map(_split_patches).apply(tf.data.experimental.unbatch())

    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = 0.08

    # Filter function which filters the dataset after total image variation.
    # See: https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/image/total_variation
    def _filter_func(sample)->bool:
        variation = tf.image.total_variation(sample[0])
        num_pixels = sample[0].get_shape().num_elements()
        var_per_pixel = (variation / num_pixels)
        return var_per_pixel > threshold

    dataset = patches_dataset.filter(lambda patch, label: _filter_func((patch, label))).take(args.num_samples).shuffle(100000)

    writer = tf.io.TFRecordWriter(args.output_dataset)

    # Make file readable for all users
    cutil.publish(args.output_dataset)

    def _encode_func(sample):
        return ctfd.encode({'patch': ctf.float_feature(sample[0].numpy().flatten()), 'label': ctf.int64_feature(sample[1].numpy())})

    # Iterate over whole dataset and write serialized examples to file.
    # See: https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/contrib/eager/Iterator
    for sample in tfe.Iterator(dataset):
        example = _encode_func(sample)
        writer.write(example.SerializeToString())

    # Flush and close the writer.
    writer.flush()
    writer.close()
        


if __name__ == "__main__":
    main(sys.argv[1:])