import os
import gc
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm


def traingmm(data_split, data_dir, model_path, model_name, output_path):

    load_bon = os.path.join(model_path, "bon" + ".gmm")
    load_sp = os.path.join(model_path, "sp" + ".gmm")
    # Load saved GMM model for bonafide and spoof
    gmm_bon = pickle.load(open(load_bon, "rb"))
    gmm_sp = pickle.load(open(load_sp, "rb"))

    gc.enable()

    filename_list = os.listdir(data_dir)
    # sort directory by lex order
    filename_list.sort()
    filepath_list = [ os.path.join(data_dir, filename) for filename in filename_list ]
    print(filename_list)

    data_collect = []
    for filepath in filepath_list:
        with open(filepath, "rb") as infile:
            data = pickle.load(infile)
            data_collect.append(data)

    bon_score = []
    sp_score = []

    if data_split == "eval":

        data = []
        for i in range(len(data_collect[0])):
            data_comb = []
            for j in range(len(data_collect)):
                data_comb.append(data_collect[j][i])
            data_stack = np.stack(data_comb, axis=1)
            data.append(data_stack)

        for i in tqdm(range(len(data))):

            bscore = gmm_bon.score(data[i])
            sscore = gmm_sp.score(data[i])

            bon_score.append(np.expand_dims(np.asarray([bscore]), axis=1))
            sp_score.append(np.expand_dims(np.asarray([sscore]), axis=1))
    else:
        data = []
        label_list = []
        for i in range(len(data_collect[0])):
            data_comb = []
            for j in range(len(data_collect)):
                data_comb.append(data_collect[j][i][0])
            data_stack = np.stack(data_comb, axis=1)
            data.append(data_stack)
            label_list.append(data_collect[0][i][1])

        for i in tqdm(range(len(data))):
            x = data[i]
            label = label_list[i]

            bscore = gmm_bon.score(x)
            sscore = gmm_sp.score(x)

            bon_score.append((np.expand_dims(np.asarray([bscore]), axis=1), label))
            sp_score.append((np.expand_dims(np.asarray([sscore]), axis=1), label))

    bon_pkl_name = "{}-bon-{}-filelevel.pkl".format(model_name, data_split)
    sp_pkl_name = "{}-sp-{}-filelevel.pkl".format(model_name, data_split)

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
        "--data_dir",
        required=True,
        type=str,
        default="./experiment/lfcc-mfcc",
        help="path to pickled file. For example, experiment/lfcc-mfcc",
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
    traingmm(args.data_split, args.data_dir, args.model_path, args.model_name, args.output_path)
