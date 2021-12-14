import os
import gc
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm


def traingmm(data_split, data_path, model_path, model_name, output_path):

    load_bon = os.path.join(model_path, "bon" + ".gmm")
    load_sp = os.path.join(model_path, "sp" + ".gmm")
    # Load saved GMM model for bonafide and spoof
    gmm_bon = pickle.load(open(load_bon, "rb"))
    gmm_sp = pickle.load(open(load_sp, "rb"))

    gc.enable()

    # Read processed train data
    with open(data_path, "rb") as infile:
        data = pickle.load(infile)
    print(len(data))
    bon_score = []
    sp_score = []

    if data_split == "eval":

        for i in tqdm(range(len(data))):

            bscore = gmm_bon.score_samples(data[i])
            sscore = gmm_sp.score_samples(data[i])

            bon_score.append(bscore)
            sp_score.append(sscore)
    else:
        for i in tqdm(range(len(data))):
            x, label = data[i]

            bscore = gmm_bon.score_samples(x)
            sscore = gmm_sp.score_samples(x)

            bon_score.append((bscore, label))
            sp_score.append((sscore, label))

    bon_pkl_name = "{}-bon-{}.pkl".format(model_name, data_split)
    sp_pkl_name = "{}-sp-{}.pkl".format(model_name, data_split)

    bon_output_path = os.path.join(output_path, data_split, bon_pkl_name)
    sp_output_path = os.path.join(output_path, data_split, sp_pkl_name)

    with open(bon_output_path, "wb") as outfile:
        pickle.dump(bon_score, outfile)
        print("dumpped", bon_output_path)

    with open(sp_output_path, "wb") as outfile:
        pickle.dump(sp_score, outfile)
        print("dumpped", sp_output_path)


if __name__ == "__main__":
    # Parser argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_split",
        required=True,
        type=str,
        help="specific which data split to process. Options [train|dev|eval]",
        default="train",
    )
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        default="./data/train/mfcc.pkl",
        help="path to pickled file. For example, data/train/mfcc.pkl",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        default="./model/mfcc",
        help="path to load model. For example, ./model/mfcc",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        default="mfcc-recur",
        help="maodel name. For example, mfcc-recur",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        default="./ensemble/mfcc",
        help="path to save ensemble preprocess data output. For example, ./ensemble/test-1",
    )
    args = parser.parse_args()

    # Create folder to store model
    if not os.path.exists(os.path.join(args.output_path, args.data_split)):
        os.makedirs(os.path.join(args.output_path, args.data_split))

    # Train
    traingmm(args.data_split, args.data_path, args.model_path, args.model_name, args.output_path)
