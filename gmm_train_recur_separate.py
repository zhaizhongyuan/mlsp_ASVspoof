import os
import gc
import time
import pickle
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def traingmm(train_path, dest):
    # Create GMM model for bonafide and spoof data
    gmm_bon = GMM(
        n_components=512,
        covariance_type="diag",
        n_init=1,
        verbose=2,
        max_iter=300,
        warm_start=True,
    )
    gmm_sp = GMM(
        n_components=512,
        covariance_type="diag",
        n_init=1,
        verbose=2,
        max_iter=20,   # set low max iter to limit converging speed
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

    data = None
    gc.collect()

    Xbon = np.vstack(bondata)
    print("Bon feature stacked, shape as: ", Xbon.shape)
    Xsp = np.vstack(spdata)
    print("Sp feature stacked, shape as: ", Xsp.shape)

    # clear mem
    bondata = None
    spdata = None
    gc.collect()

    # iteration through chunks of data
    num_chunk = 10
    sp_chunk_size = (Xsp.shape[0] // num_chunk) + 1

    # iteration of chunks
    for j in range(num_chunk):

        print("training on chunk {}".format(j))


        # Train spoof
        print("gmm sp training on chunk from {} to {} of size {}".format(j*sp_chunk_size, (j+1)*sp_chunk_size, sp_chunk_size))
        gmm_sp.fit(Xsp[j*sp_chunk_size : (j+1)*sp_chunk_size])

    pickle.dump(
        gmm_sp,
        open(os.path.join(dest, "sp" + ".gmm"), "wb"),
    )
    print("GMM sp model created")

    print("gmm bon training on all")
    gmm_bon.fit(Xbon)

    # save model after iteration
    pickle.dump(
        gmm_bon,
        open(os.path.join(dest, "bon" + ".gmm"), "wb"),
    )
    print("GMM bon model created")


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
