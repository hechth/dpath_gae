import sys, argparse, json
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor

import packages.Tensorflow.Image as ctfi
import packages.Utility as cutil
tf.enable_eager_execution()

import packages.Tensorflow.Model.SavedModel as ctfsm

def main(argv):
    parser = argparse.ArgumentParser(description='Compute latent code for image patch by model inference.')

    parser.add_argument('export_dir',type=str,help='Path to saved model to use for inference.')
    parser.add_argument('filename', type=str,help='Image file or numpy array to run inference on.')
    parser.add_argument('--output', type=str, help='Where to store the output.')

    args = parser.parse_args()
        
    predict_fn = predictor.from_saved_model(args.export_dir)

    # Extract patch size and latent space size from the model identifier
    patch_size = ctfsm.determine_patch_size(args.export_dir)
    latent_space_size = ctfsm.determine_latent_space_size(args.export_dir)

    image = None
    
    # Check if it is image or numpy array data
    if ctfi.is_image(args.filename):
        image = ctfi.load(args.filename).numpy()
    elif cutil.is_numpy_format(args.filename):
        image = np.load(args.filename)
    else:
        sys.exit(3)

    # Resize image to match size required by the model
    image = np.resize(image, [patch_size, patch_size, 3])

    batch = np.expand_dims(image,0)
    # Make predictions
    pred = predict_fn({
        'fixed': batch,
        'moving': np.random.rand(1, patch_size,patch_size,3),
        'embedding': np.random.rand(1,1,1,latent_space_size)
    })
    latent_code = pred['latent_code_fixed']
    print(latent_code)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'filename': args.filename, 'model': args.export_dir, 'latent_code': latent_code.tolist() }, f)
    

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main, argv=sys.argv[1:])