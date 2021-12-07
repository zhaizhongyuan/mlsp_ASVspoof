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
    # print("y pred max", y_diff.max(), " y pred min", y_diff.min())
    k = 1.0
    y_prob = 1.0 / (1.0 + np.exp(-k * y_diff))
    return y_prob

def eer_score(y_true, y_pred):
    fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true, y_pred)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    # print("y pred max", y_pred.max(), " y pred min", y_pred.min())

    fig,ax = plt.subplots()
    ax.plot(fpr,tpr)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve")
    plt.show()

    return eer

def testgmm(test_path, dest_bon, dest_sp, label_path, output_path):
    # Load saved GMM model for bonafide and spoof
    gmm_bon = pickle.load(open(dest_bon, "rb"))
    gmm_sp = pickle.load(open(dest_sp, "rb"))

    with open(test_path, "rb") as infile:
        data = pickle.load(infile)
    print(len(data))

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
    data_path = args.data_path
    dest_bon = args.model_path_bon
    dest_sp = args.model_path_sp
    label_path = args.label_path
    output_path = args.output_path

    # Test
    testgmm(data_path, dest_bon, dest_sp, label_path, output_path)
