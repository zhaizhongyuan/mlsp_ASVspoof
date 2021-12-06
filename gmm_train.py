from genericpath import exists
import os
import numpy as np
import pickle

# import soundfile as sf
# import librosa
import time
import gc
import argparse

# from python_speech_features import mfcc
from sklearn.mixture import GaussianMixture as GMM

# from sklearn import preprocessing
# from pdb import set_trace
# from scipy import stats
def traingmm(train_path, dest):
    gmm_bon = GMM(
        n_components=512,
        covariance_type="diag",
        n_init=1,
        verbose=2,
        max_iter=600,
        warm_start=True,
    )
    gmm_sp = GMM(
        n_components=512,
        covariance_type="diag",
        n_init=1,
        verbose=2,
        max_iter=600,
        warm_start=True,
    )
    gc.enable()

    # Train each 10 piece of data
    bondata = []
    spdata = []

    with open(train_path, "rb") as infile:
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

    Xbon = np.vstack(bondata)
    print("Bon feature stacked, shape as: ", Xbon.shape)
    Xsp = np.vstack(spdata)
    print("Sp feature stacked, shape as: ", Xsp.shape)

    # clear mem
    # gc.enable()
    bondata = None
    spdata = None
    gc.collect()

    t0 = time.time()
    gmm_bon.fit(Xbon)
    print("Bon gmm trained, time spend:", time.time() - t0)
    pickle.dump(
        gmm_bon,
        open(os.path.join(dest, "bon" + ".gmm"), "wb"),
    )
    print("GMM bon model created")

    # clear mem
    # gmm_bon = None
    Xbon = None
    gc.collect()

    half = Xsp.shape[0] // 2

    print("Sp gmm train first half", half)
    t0 = time.time()
    gmm_sp.fit(Xsp[:half])
    print("Sp gmm trained, time spend:", time.time() - t0)

    print("Sp gmm train second half", Xsp.shape[0] - half)
    t0 = time.time()
    gmm_sp.fit(Xsp[half:])
    print("Sp gmm trained, time spend:", time.time() - t0)

    pickle.dump(
        gmm_sp,
        open(os.path.join(dest, "sp" + ".gmm"), "wb"),
    )
    print("GMM sp model created")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=False,
        type=str,
        default="./data/train/mfcc.pkl",
        help="path to pickled file. For example, data/train.pkl",
    )
    parser.add_argument(
        "--model_path",
        required=False,
        type=str,
        default="./model/mfcc",
        help="path to save model. For example, ./model/mfcc",
    )
    args = parser.parse_args()

    train_path = args.data_path
    dest = args.model_path

    if not os.path.exists(dest):
        os.makedirs(dest)
    traingmm(train_path, dest)
