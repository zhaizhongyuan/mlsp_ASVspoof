import os
import gc
import time
import pickle
import argparse
import numpy as np
# import librosa
# import soundfile as sf
from sklearn.mixture import GaussianMixture as GMM


def traingmm(train_path, dest, feature_type):
    gmm_bon = GMM(n_components = 512, covariance_type='diag',n_init = 1,verbose=2, max_iter=600, warm_start=True) 
    gmm_sp  = GMM(n_components = 512, covariance_type='diag',n_init = 1,verbose=2, max_iter=600, warm_start=True)
    gc.enable()

    # Train each 10 piece of data
    for num in range(10):
        bondata = []
        spdata = []

        filename = train_path + "-{}.pkl".format(num)
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

        Xbon = np.vstack(bondata)
        print('Bon feature stacked, shape as: ', Xbon.shape)
        Xsp = np.vstack(spdata)
        print('Sp feature stacked, shape as: ', Xsp.shape)

        # clear mem
        # gc.enable()
        bondata = None
        spdata = None
        gc.collect()

        t0 = time.time()
        gmm_bon.fit(Xbon)
        print('Bon gmm trained, time spend:', time.time() - t0)
        pickle.dump(gmm_bon, open(dest + 'bon'+ "_epoch{}".format(num) + '.gmm', 'wb'))
        print('GMM bon model created')

        # clear mem
        # gmm_bon = None
        Xbon = None
        gc.collect()

        half = Xsp.shape[0] // 2

        print("Sp gmm train first half", half)
        t0 = time.time()
        gmm_sp.fit(Xsp[:half])
        print('Sp gmm trained, time spend:', time.time() - t0)

        print("Sp gmm train second half", Xsp.shape[0]-half)
        t0 = time.time()
        gmm_sp.fit(Xsp[half:])
        print('Sp gmm trained, time spend:', time.time() - t0)

        pickle.dump(gmm_sp, open(dest + 'sp' + "_epoch{}".format(num) + '.gmm', 'wb'))
        print('GMM sp model created')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str,  default='./data/train.pkl', help='path to pickled file. For example, data/train.pkl')
    parser.add_argument("--model_path", required=True, type=str, default='./model/', help='path to pickled file. For example, data/train.pkl')
    parser.add_argument("--feature_type", required=True, type=str, default='lfcc', help='select the feature type. lfcc or mfcc')
    args = parser.parse_args()

    train_path = args.data_path
    dest = args.model_path

    traingmm(train_path, dest, args.feature_type)