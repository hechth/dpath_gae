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
    parser.add_argument('--no_filter', dest='no_filter', action='store_true', default=False, help='Whether to apply total image variation filtering.')
    parser.add_argument('--threshold', type=float, dest='threshold', help='Threshold for filtering the samples according to variation.')
    parser.add_argument('--subsampling_factor',type=int, dest='subsampling_factor', default=1, help='Subsampling factor to use to downsample images.')
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


    if args.subsampling_factor > 1:
        images_dataset = images_dataset.map(lambda feature: {'image': ctfi.subsample(feature['image'], args.subsampling_factor), 'label': feature['label']})

    def _filter_func_label(features):
        label = features['label']
        result = label > -1
        return result
        
    images_dataset = images_dataset.filter(_filter_func_label).shuffle(100)
    # Extract image patches

    #for sample in tfe.Iterator(images_dataset):
    #    print(sample['label'])

    def _split_patches(features):
        patches = ctfi.extract_patches(features['image'], args.patch_size)
        labels = tf.expand_dims(tf.reshape(features['label'], [1]),0)
        labels = tf.tile(labels,tf.stack([tf.shape(patches)[0], 1]))
        return (patches, labels)

    patches_dataset = images_dataset.map(_split_patches).apply(tf.data.experimental.unbatch())

    patches_dataset = patches_dataset.map(lambda patch, label: {'patch': patch, 'label': label})

    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = 0.08


    num_filtered_patches = tf.Variable(0)

    # Filter function which filters the dataset after total image variation.
    # See: https://www.tensorflow.org/versions/r1.12/api_docs/python/tf/image/total_variation
    def add_background_info(sample):
        variation = tf.image.total_variation(sample['patch'])
        num_pixels = sample['patch'].get_shape().num_elements()
        var_per_pixel = (variation / num_pixels)
        no_background = var_per_pixel > threshold

        def true_fn():
             sample.update({'no_background': True})
             return sample

        def false_fn():
            def _true_fn_lvl2():
                sample.update({'label':tf.reshape(tf.convert_to_tensor(len(args.labels), dtype=tf.int64), [1]),'no_background': True})
                return sample

            def _false_fn_lvl2():
                sample.update({'no_background': False})
                return sample

            pred = tf.equal(num_filtered_patches.value() % 10, 0)
            num_filtered_patches.assign_add(1)            
            return tf.cond(pred,true_fn=_true_fn_lvl2,false_fn=_false_fn_lvl2)       
        return tf.cond(no_background,true_fn=true_fn, false_fn=false_fn)

    
    if args.no_filter == True:
        dataset = patches_dataset
    else:
        dataset = patches_dataset.map(add_background_info).filter(lambda sample: sample['no_background'])
    
    dataset = dataset.map(lambda sample: (sample['patch'], sample['label']))
    dataset = dataset.take(args.num_samples).shuffle(100000)

    writer = tf.io.TFRecordWriter(args.output_dataset)

    # Make file readable for all users
    cutil.publish(args.output_dataset)

    def _encode_func(sample):
        patch_np = sample[0].numpy().flatten()
        label_np = sample[1].numpy()
        return ctfd.encode({'patch': ctf.float_feature(patch_np), 'label': ctf.int64_feature(label_np)})

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