import os
import gc
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm


def testgmm(test_path, dest_bon, dest_sp):
    # Load saved GMM model for bonafide and spoof
    gmm_bon = pickle.load(open(dest_bon, "rb"))
    gmm_sp = pickle.load(open(dest_sp, "rb"))

    # Read processed dev/test data
    bondata = []
    spdata = []
    with open(test_path, "rb") as infile:
        data = pickle.load(infile)
        for t in data:
            if t is None:
                continue
            feats, label = t

            # label selection
            if label == "bonafide":
                bondata.append(feats)
            elif label == "spoof":
                spdata.append(feats)
    print(len(bondata), bondata[0].shape)
    print(len(spdata), spdata[0].shape)

    # Make predictions
    predb_raw = []
    preds_raw = []
    j_bon = len(bondata)
    k_sp = len(spdata)

    # Bonafide
    for i in tqdm(range(j_bon)):
        X = bondata[i]
        bscore = gmm_bon.score(X)
        sscore = gmm_sp.score(X)
        predb_raw.append(bscore - sscore)

    # Spoof
    for i in tqdm(range(k_sp)):
        X = spdata[i]
        bscore = gmm_bon.score(X)
        sscore = gmm_sp.score(X)
        preds_raw.append(bscore - sscore)

    # Calculate accuracy
    predb = np.asarray(predb_raw)
    preds = np.asarray(preds_raw)
    predb[predb < 0] = 0
    predb[predb > 0] = 1
    predb_result = np.sum(predb)
    print(
        int(predb_result),
        "Bon samples were CORRECTLY evaluated out of",
        j_bon,
        "samples. Bon_Accuracy = ",
        predb_result / j_bon,
    )

    preds[preds > 0] = 0
    preds[preds < 0] = 1
    preds_result = np.sum(preds)
    print(
        int(preds_result),
        "Sp samples were CORRECTLY evaluated out of",
        k_sp,
        "samples. Sp_Accuracy = ",
        preds_result / k_sp,
    )

    print(
        "Total GMM Classifier Accuracy = ",
        (predb_result + preds_result) / (j_bon + k_sp),
    )


if __name__ == "__main__":
    # Parser Argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        default="./data/dev/mfcc.pkl",
        help="path to pickled file. For example, ./data/dev/mfcc.pkl",
    )
    parser.add_argument(
        "--model_path_bon",
        required=True,
        type=str,
        default="./model/mfcc/bon.gmm",
        help="path to pickled file. For example, ./model/mfcc/bon.gmm",
    )
    parser.add_argument(
        "--model_path_sp",
        required=True,
        type=str,
        default="./data/mfcc/sp.gmm",
        help="path to pickled file. For example, ./data/mfcc/sp.gmm",
    )
    args = parser.parse_args()
    dev_path = args.data_path
    dest_bon = args.model_path_bon
    dest_sp = args.model_path_sp

    # Test
    testgmm(dev_path, dest_bon, dest_sp)
