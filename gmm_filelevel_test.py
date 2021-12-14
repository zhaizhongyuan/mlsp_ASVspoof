import os
import gc
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sklearn.metrics

def sigmoid(y_diff):
    k = 1.0
    y_prob = 1.0 / (1.0 + np.exp(-k * y_diff))
    return y_prob

def eer_score(y_true, y_pred):
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true, y_pred)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    fig,ax = plt.subplots()
    ax.plot(fpr,tpr)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve")
    plt.show()

    return eer

def testgmm(test_dir, dest_bon, dest_sp, label_path, output_path):
    # Load saved GMM model for bonafide and spoof
    gmm_bon = pickle.load(open(dest_bon, "rb"))
    gmm_sp = pickle.load(open(dest_sp, "rb"))

    filename_list = os.listdir(test_dir)
    # sort directory by lex order
    filename_list.sort()
    filepath_list = [ os.path.join(test_dir, filename) for filename in filename_list ]
    print(filename_list)

    data_collect = []
    for filepath in filepath_list:
        with open(filepath, "rb") as infile:
            data_instance = pickle.load(infile)
            data_collect.append(data_instance)

    data = []
    for i in range(len(data_collect[0])):
        data_comb = []
        for j in range(len(data_collect)):
            data_comb.append(data_collect[j][i])
        data_stack = np.concatenate(data_comb, axis=1)
        data.append(data_stack)


    with open(label_path, "r") as f:
        lines = f.readlines()

    result_text = []
    y_true = []
    y_diff = []
    for i in tqdm(range(len(lines))):
        line = lines[i].rstrip()
        label = line.split()[-1]
        if label == "spoof":
            y_true.append(0)
        else:
            y_true.append(1)

        bscore = gmm_bon.score(data[i])
        sscore = gmm_sp.score(data[i])

        diff_score = bscore - sscore
        y_diff.append(diff_score)

        result_text.append(line.rstrip() + " {:.6f}\n".format(diff_score))

    with open(output_path, "w") as file:
        file.writelines(result_text)

    y_pred = sigmoid( np.asarray(y_diff) )
    y_true = np.asarray( y_true )

    eer = eer_score(y_true, y_pred)
    print('Total GMM Classifier EER = ', eer)


if __name__ == "__main__":
    # Parser Argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        required=True,
        type=str,
        default="./experiment/lfcc/eval",
        help="path to data directory. For example, ./experiment/lfcc/eval",
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
    parser.add_argument(
        "--label_path",
        required=True,
        type=str,
        help="path to label file. For example, LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        default="/mnt/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        default="./output/eval-mfcc-recur-1.txt",
        help="path to output. For example, ./output/eval-mfcc-recur-1.txt",
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    dest_bon = args.model_path_bon
    dest_sp = args.model_path_sp
    label_path = args.label_path
    output_path = args.output_path

    # Test
    testgmm(data_dir, dest_bon, dest_sp, label_path, output_path)
