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

def testgmm(test_path, dest_bon, dest_sp, feature_type):
    # training data accuracy
    gmm_bon = pickle.load(open(dest_bon,'rb'))
    gmm_sp  = pickle.load(open(dest_sp,'rb'))

    bondata = []
    spdata = []

    for num in range(10):
        filename = test_path + "-{}.pkl".format(num)
        with open(filename, 'rb') as infile:
            data = pickle.load(infile)
            for t in data:
                if t is None:
                    continue
                feat_lfcc, feat_mfcc, label = t

                # feature selection
                if feature_type == "lfcc":
                    feats = feat_lfcc
                elif feature_type == "mfcc":
                    feats = feat_mfcc

                # label selection
                if (label == 'bonafide'):
                    bondata.append(feats)
                elif(label == 'spoof'):
                    spdata.append(feats)

    print(len(bondata), bondata[0].shape)
    print(len(spdata), spdata[0].shape)

    predb = []
    preds = []
    j_bon = len(bondata)
    k_sp  = len(spdata)


    for i in tqdm(range(j_bon)):
        X = bondata[i]
        bscore = gmm_bon.score(X)
        sscore = gmm_sp.score(X)
        predb.append(bscore-sscore)

    for i in tqdm(range(k_sp)):
        X = spdata[i]
        bscore = gmm_bon.score(X)
        sscore = gmm_sp.score(X)
        preds.append(bscore-sscore)

    predb1 = np.asarray(predb)
    preds1 = np.asarray(preds)

    predb1[predb1 < 0] = 0
    predb1[predb1 > 0] = 1
    predbresult1 = np.sum(predb1)
    print(predbresult1, 'Bon samples were CORRECTLY evaluated out of', j_bon,'samples. Bon_Accuracy = ', predbresult1/j_bon )# 0.7356

    preds1[preds1 > 0] = 0
    preds1[preds1 < 0] = 1
    predsresult = np.sum(preds1)
    print(predsresult, 'Sp samples were CORRECTLY evaluated out of', k_sp, 'samples. Sp_Accuracy = ', predsresult/k_sp)# 0.4092

    print('Total GMM Classifier Accuracy = ',(predbresult1 + predsresult)/(j_bon + k_sp))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str,  default='./data/dev', help='path to pickled file. For example, ./data/dev')
    parser.add_argument("--model_path_bon", required=True, type=str, default='./model/mfcc_gmm_bon_epoch9.gmm', help='path to pickled file. For example, ./model/mfcc_gmm_bon_epoch9.gmm')
    parser.add_argument("--model_path_sp", required=True, type=str, default='./data/mfcc_gmm_sp_epoch9.gmm', help='path to pickled file. For example, ./data/mfcc_gmm_sp_epoch9.gmm')
    parser.add_argument("--feature_type", required=True, type=str, default='mfcc', help='select the feature type. lfcc or mfcc')
    args = parser.parse_args()

    dev_path = args.data_path
    dest_bon = args.model_path_bon
    dest_sp = args.model_path_sp

    testgmm(dev_path, dest_bon, dest_sp, args.feature_type)