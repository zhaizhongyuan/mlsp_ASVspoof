import os
import gc
import time
import pickle
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture as GMM


def traingmm(train_path, dest):
    # Create GMM model for bonafide and spoof data
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

    # Read processed train data
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
    bondata = None
    spdata = None
    gc.collect()

    # Train bonafide
    t0 = time.time()
    print("Train on Xbon shape {}".format(Xbon.shape[0]))
    gmm_bon.fit(Xbon)
    print("Bon gmm trained, time spend:", time.time() - t0)
    pickle.dump(
        gmm_bon,
        # open(os.path.join(dest, "bon" + ".gmm"), "wb"),
        open(os.path.join(dest, "bon_fit_all_partial" + ".gmm"), "wb"),
    )
    print("GMM bon model created")

    # clear memory
    gmm_bon = None
    Xbon = None
    gc.collect()

    # Train spoof
    Xsp_first_partial = Xsp[:Xsp.shape[0] // 2]
    Xsp_second_partial = Xsp[Xsp.shape[0] // 2:]
    t0 = time.time()
    print("Train on Xsp first partial shape {}".format(Xsp_first_partial.shape[0]))
    gmm_sp.fit(Xsp_first_partial)
    print("First half Sp gmm trained, time spend:", time.time() - t0)
    t0 = time.time()
    print("Train on Xsp second partial shape {}".format(Xsp_second_partial.shape[0]))
    gmm_sp.fit(Xsp_second_partial)
    print("Sp gmm trained, time spend:", time.time() - t0)
    pickle.dump(
        gmm_sp,
        # open(os.path.join(dest, "sp" + ".gmm"), "wb"),
        open(os.path.join(dest, "spoof_fit_all_partial" + ".gmm"), "wb"),
    )
    print("GMM sp model created")


if __name__ == "__main__":
    # Parser argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=False,
        type=str,
        default="./data/train/mfcc.pkl",
        help="path to pickled file. For example, data/train/mfcc.pkl",
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

    # Create folder to store model
    if not os.path.exists(dest):
        os.makedirs(dest)

    # Train
    traingmm(train_path, dest)
