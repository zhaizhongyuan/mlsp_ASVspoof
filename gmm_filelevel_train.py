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


def traingmm(train_dir, dest, load_model):
    if not load_model:
        # Create GMM model for bonafide and spoof data
        gmm_bon = GMM(
            n_components=64,
            covariance_type="diag",
            n_init=1,
            verbose=2,
            max_iter=300,
            warm_start=True,
        )
        gmm_sp = GMM(
            n_components=64,
            covariance_type="diag",
            n_init=1,
            verbose=2,
            max_iter=300,
            warm_start=True,
        )
    else:
        load_bon = os.path.join(load_model, "bon" + ".gmm")
        load_sp = os.path.join(load_model, "sp" + ".gmm")
        # Load saved GMM model for bonafide and spoof
        gmm_bon = pickle.load(open(load_bon, "rb"))
        gmm_sp = pickle.load(open(load_sp, "rb"))

    gc.enable()

    filename_list = os.listdir(train_dir)
    # sort directory by lex order
    filename_list.sort()
    filepath_list = [ os.path.join(train_dir, filename) for filename in filename_list ]
    print(filename_list)

    data_collect = []
    for filepath in filepath_list:
        with open(filepath, "rb") as infile:
            data = pickle.load(infile)
            data_collect.append(data)

    bondata = []
    spdata = []
    for i in range(len(data_collect[0])):
        data_comb = []
        for j in range(len(data_collect)):
            data_comb.append(data_collect[j][i][0])
        data_stack = np.concatenate(data_comb, axis=1)

        if data_collect[0][i][1] == "bonafide":
            bondata.append(data_stack)
        elif data_collect[0][i][1] == "spoof":
            spdata.append(data_stack)
        else:
            raise ValueError('label error')

    data_collect = None
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

    print("gmm sp training on all")
    gmm_sp.fit(Xsp)

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
        "--data_dir",
        required=True,
        type=str,
        default="./experiment/test",
        help="path to all pickled score file. For example, experiment/test",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        default="./model/mfcc",
        help="path to save model. For example, ./model/mfcc",
    )
    parser.add_argument(
        "--load_model_path",
        required=False,
        type=str,
        default="",
        help="path to load previous model for continue training, default is none. For example, ./model/mfcc",
    )
    args = parser.parse_args()
    train_dir = args.data_dir
    dest = args.model_path
    load_model = args.load_model_path

    # Create folder to store model
    if not os.path.exists(dest):
        os.makedirs(dest)

    # Train
    traingmm(train_dir, dest, load_model)
