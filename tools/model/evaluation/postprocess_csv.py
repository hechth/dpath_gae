import sys, argparse, os, math
import git
git_root = git.Repo('.', search_parent_directories=True).working_tree_dir
sys.path.append(git_root)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
from ast import literal_eval as make_tuple

fieldnames = ["method", "landmark", "min_idx", "min_idx_value", "rank", "landmark_value"]

def get_landmarks(filename, subsampling_factor=1):
    landmarks = []
    with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                landmarks.append([float(row[2]) / subsampling_factor, float(row[1]) / subsampling_factor])
            landmarks = np.array(landmarks,dtype=np.float32)
    return landmarks

def main(argv):
    parser = argparse.ArgumentParser(description='Compute codes and reconstructions for image.')
    parser.add_argument('csv_filename', type=str,help='CSV file containing the results.')
    parser.add_argument('source_landmarks', type=str,help='CSV file from which to extract the landmarks for source image.')
    parser.add_argument('target_landmarks', type=str,help='CSV file from which to extract the landmarks for target image.')
    parser.add_argument('patch_size', type=int, help='Size of image patch.')
    parser.add_argument('region_size', type=int)
    parser.add_argument('output_csv', type=str, help='Filename for postprocessed output file')
    args = parser.parse_args()

    offset = [args.region_size / 2, args.region_size / 2]

    data = []

    with open(args.csv_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames)
        data = [row for row in reader]
    header = data.pop(0)

    # Get source and target landmarks
    source_landmarks = get_landmarks(args.source_landmarks, 1)
    target_landmarks = get_landmarks(args.target_landmarks, 1)

    with open(args.output_csv,'wt') as outfile:
        for record in data:
            landmark_idx = int(record["landmark"])
            processed_record = {"pos_src_yx" : source_landmarks[landmark_idx], "pos_tgt_yx" : target_landmarks[landmark_idx]}
            min_idx = np.array(make_tuple(record["min_idx"]))
            min_idx_pos = processed_record["pos_tgt_yx"] + (min_idx - offset)
            processed_record["min_idx_pos_yx"] = min_idx_pos
            processed_record["distance"] = np.linalg.norm(processed_record["pos_tgt_yx"] - processed_record["min_idx_pos_yx"])

            record.update(processed_record)
        

        all_fieldnames = data[0].keys()
        fp = csv.DictWriter(outfile, all_fieldnames)
        fp.writeheader()
        fp.writerows(data)


    return 0
        


if __name__ == "__main__":
    main(sys.argv[1:])