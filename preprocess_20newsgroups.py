import argparse
from dataset import preprocess_20newsgroups, apply_transformation_20newsgroups
import os
import shutil

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="20news-bydate-train/, 20news-bydate-test/ and vocabulary.txt are located in this directory.")
    parser.add_argument("--threshold", type=int, default=512, help="Hyperparameter indicates the maximum number of tokens per new.")
    args = parser.parse_args()

    dir_20_full = os.path.join(args.data_dir, "20_full")
    preprocess_20newsgroups(args.data_dir, os.path.join(args.data_dir, "vocabulary.txt"), dir_20_full) # some files are ignored
    apply_transformation_20newsgroups(args.data_dir, threshold=args.threshold)

if __name__ == "__main__": 
    main()  