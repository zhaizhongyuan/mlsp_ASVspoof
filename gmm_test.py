import os
import gc
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm

# import librosa
# import soundfile as sf
from sklearn.mixture import GaussianMixture as GMM

import argparse


def testgmm(test_path, dest_bon, dest_sp):
    # training data accuracy
    gmm_bon = pickle.load(open(dest_bon, "rb"))
    gmm_sp = pickle.load(open(dest_sp, "rb"))

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

    predb = []
    preds = []
    j_bon = len(bondata)
    k_sp = len(spdata)

    for i in tqdm(range(j_bon)):
        X = bondata[i]
        bscore = gmm_bon.score(X)
        sscore = gmm_sp.score(X)
        predb.append(bscore - sscore)

    for i in tqdm(range(k_sp)):
        X = spdata[i]
        bscore = gmm_bon.score(X)
        sscore = gmm_sp.score(X)
        preds.append(bscore - sscore)

    predb1 = np.asarray(predb)
    preds1 = np.asarray(preds)

    predb1[predb1 < 0] = 0
    predb1[predb1 > 0] = 1
    predbresult1 = np.sum(predb1)
    print(
        predbresult1,
        "Bon samples were CORRECTLY evaluated out of",
        j_bon,
        "samples. Bon_Accuracy = ",
        predbresult1 / j_bon,
    )  # 0.7356

    preds1[preds1 > 0] = 0
    preds1[preds1 < 0] = 1
    predsresult = np.sum(preds1)
    print(
        predsresult,
        "Sp samples were CORRECTLY evaluated out of",
        k_sp,
        "samples. Sp_Accuracy = ",
        predsresult / k_sp,
    )  # 0.4092

    print(
        "Total GMM Classifier Accuracy = ",
        (predbresult1 + predsresult) / (j_bon + k_sp),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        default="./data/dev/mfcc.pkl",
        help="path to pickled file. For example, ./data/dev",
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
    testgmm(dev_path, dest_bon, dest_sp)
