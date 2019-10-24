import sys, getopt
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
    # Get folder for saved model and image path
    export_dir = ''
    input_filename = ''

    try:
        opts, args = getopt.getopt(argv,"hm:i:",["model_path=","input="])
    except getopt.GetoptError:
        print('Usage --model <model_path> --input <input_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h","-help","help","--help"):
            print('Usage --model <model_path> --input <input_path>')
            sys.exit()
        elif opt in ("--model", "--model_path", "--m", "-m"):
            export_dir = arg
        elif opt in ("--input", "--filename", "--i", "-i"):
            input_filename = arg
        
    predict_fn = predictor.from_saved_model(export_dir)

    # Extract patch size and latent space size from the model identifier
    patch_size = ctfsm.determine_patch_size(export_dir)
    latent_space_size = ctfsm.determine_latent_space_size(export_dir)

    image = None
    
    # Check if it is image or numpy array data
    if ctfi.is_image(input_filename):
        image = ctfi.load(input_filename).numpy()
    elif cutil.is_numpy_format(input_filename):
        image = np.load(input_filename)
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
    })['latent_code_fixed']
    print(pred)
    

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main, argv=sys.argv[1:])