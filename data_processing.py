import os
import numpy as np
import pickle
import argparse
import soundfile as sf
import multiprocessing
import spafe.features.lfcc
import spafe.features.mfcc
from ctypes import c_int
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=False, type=str, help='path to ASVSpoof data directory. For example, LA/ASVspoof2019_LA_train/flac/', default="/mnt/LA/ASVspoof2019_LA_train/flac/")
parser.add_argument("--label_path", required=False, type=str, help='path to label file. For example, LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', default="/mnt/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
parser.add_argument("--output_path", required=False, type=str, help='path to output pickle file. For example, ./data/train.pkl', default="/home/yuxuan/MLSP_ASVspoof/data/train")
parser.add_argument("--ftype", required=True, type=str, help="type of featre. For example, lfcc, mfcc, silence, ...")
# parser.add_argument("--feature_type", required=True, type=str, help='select the feature type.')
args = parser.parse_args()


## modify data processing to calculate coefficient: MFCC, delta MFCC, and delta delta MFCC
def calculate_delta(array):
    rows,cols = array.shape
    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows -1:
                second = rows -1
            else:
                second = i+j
            index.append((second,first))
            j+=1
        deltas[i] = (array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]]))) / 10
    return deltas

def process_audio(filepath, feature_type):
    # read in labels
    filename2label = {}
    for line in open(args.label_path):
        line = line.split()
        filename, label = line[1], line[-1]
        filename2label[filename] = label

    filename = filepath.split('.')[0]
    if filename not in filename2label:
         # we skip speaker enrollment stage
        return
    label = filename2label[filename]
    sig, rate = sf.read(os.path.join(args.data_path, filepath))

    if feature_type == "mfcc":
        # extract lfcc, delta, delta delta
        feat_lfcc = spafe.features.lfcc.lfcc(sig, fs=rate, num_ceps=20, pre_emph=0, win_len=0.03, win_hop=0.015, nfilts=70, nfft=1024)
        delta_feat_lfcc = calculate_delta(feat_lfcc)
        delta_delta_feat_lfcc = calculate_delta(delta_feat_lfcc)
        feat = np.hstack((feat_lfcc, delta_feat_lfcc, delta_delta_feat_lfcc))
    elif feature_type == "lfcc":
        # extract mfcc, delta, delta delta
        feat_mfcc = spafe.features.mfcc.mfcc(sig, fs=rate, num_ceps=20, pre_emph=0, win_len=0.03, win_hop=0.015, nfilts=70, nfft=1024)
        delta_feat_mfcc = calculate_delta(feat_mfcc)
        delta_delta_feat_mfcc = calculate_delta(delta_feat_mfcc)
        feat = np.hstack((feat_mfcc, delta_feat_mfcc, delta_delta_feat_mfcc))
    else:
        print("Bad feature type!")
    return feat, label


if __name__ == '__main__':
    chunksize = 2600
    total_size = len(os.listdir(args.data_path)) 

    # Divide data into 10 part, but still single thread
    for n in range(10):
        feats = []
        for filepath in tqdm(os.listdir(args.data_path)[n*chunksize : (n+1)*chunksize]):
            feats.append(process_audio(filepath, args.ftype))
            # Save output data
        output_path = args.output_path + "-{}.pkl".format(n)
        with open(output_path, 'wb') as outfile:
            pickle.dump(feats, outfile)
            print("dumpped", output_path)