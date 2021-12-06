import os
import numpy as np
import pickle
import argparse
import soundfile as sf
import spafe.features.lfcc
import spafe.features.mfcc
import spafe.features.bfcc
import spafe.features.gfcc
import spafe.features.ngcc
import spafe.features.pncc
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to ASVSpoof data directory. For example, LA/ASVspoof2019_LA_train/flac/', default="/mnt/LA/ASVspoof2019_LA_train/flac/")
parser.add_argument("--label_path", required=True, type=str, help='path to label file. For example, LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', default="/mnt/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt")
parser.add_argument("--output_path", required=True, type=str, help='path to output pickle file. For example, ./data/train.pkl', default="/home/yuxuan/MLSP_ASVspoof/data/train")
parser.add_argument("--ftype", required=True, type=str, help="type of feature. For example, lfcc, mfcc, silence, ...")
args = parser.parse_args()


# modify data processing to calculate coefficient: MFCC, delta MFCC, and delta delta MFCC
def calculate_delta(array):
    # https://github.com/MohamadMerchant/Voice-Authentication-and-Face-Recognition
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

    # Read audio file
    filename = filepath.split('.')[0]
    if filename not in filename2label:
         # we skip speaker enrollment stage
        return
    label = filename2label[filename]
    sig, rate = sf.read(os.path.join(args.data_path, filepath))

    # extract feature, delta of feature, delta delta of feature
    if feature_type == "lfcc":
        feat = spafe.features.lfcc.lfcc(sig, fs=rate, num_ceps=20, pre_emph=0, win_len=0.03, win_hop=0.015, nfilts=70, nfft=1024)
    elif feature_type == "mfcc":
        feat = spafe.features.mfcc.mfcc(sig, fs=rate, num_ceps=20, pre_emph=0, win_len=0.03, win_hop=0.015, nfilts=70, nfft=1024)
    elif feature_type == "bfcc":
        feat = spafe.features.bfcc.bfcc(sig, fs=rate, num_ceps=20, pre_emph=0, win_len=0.03, win_hop=0.015, nfilts=70, nfft=1024)
    elif feature_type == "gfcc":
        feat = spafe.features.gfcc.gfcc(sig, fs=rate, num_ceps=20, pre_emph=0, win_len=0.03, win_hop=0.015, nfilts=70, nfft=1024)
    elif feature_type == "ngcc":
        feat = spafe.features.ngcc.ngcc(sig, fs=rate, num_ceps=20, pre_emph=0, win_len=0.03, win_hop=0.015, nfilts=70, nfft=1024)
    elif feature_type == "pncc":
        feat = spafe.features.pncc.pncc(sig, fs=rate, num_ceps=20, pre_emph=0, win_len=0.03, win_hop=0.015, nfilts=70, nfft=1024)
    else:
        print("Bad feature type!")
    delta_feat = calculate_delta(feat)
    delta_delta_feat = calculate_delta(delta_feat)
    combined_feat = np.hstack((feat, delta_feat, delta_delta_feat))

    return combined_feat, label


if __name__ == '__main__':
    feat_label = []
    for filepath in tqdm(os.listdir(args.data_path)):
        feat_label.append(process_audio(filepath, args.ftype))

    # Create folder to save data
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Save data
    output_path = os.path.join(args.output_path, args.ftype+".pkl")
    with open(output_path, 'wb') as outfile:
        pickle.dump(feat_label, outfile)
        print("dumpped", output_path)