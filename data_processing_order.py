import os
import time
import numpy as np
import pickle
import argparse
import soundfile as sf
import multiprocessing
import spafe.features.lfcc
import spafe.features.mfcc
import spafe.features.bfcc
import spafe.features.gfcc
import spafe.features.ngcc
from tqdm import tqdm
import silence_measure
from ctypes import c_int


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_split",
    required=True,
    type=str,
    help="specific which data split to process. Options [train|dev|eval]",
    default="train",
)
parser.add_argument(
    "--data_path",
    required=True,
    type=str,
    help="path to ASVSpoof data directory. For example, LA/ASVspoof2019_LA_train/flac/",
    default="/mnt/LA/ASVspoof2019_LA_train/flac/",
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
    help="path to output pickle file. For example, ./data/train.pkl",
    default="/home/yuxuan/MLSP_ASVspoof/data/train",
)
parser.add_argument(
    "--ftype",
    required=True,
    type=str,
    help="type of feature. For example, lfcc, mfcc, silence, ...",
)
args = parser.parse_args()

pbar = None

def increment():
    # Multiprocess counter
    with counter_lock:
        counter.value += 1
        # if counter.value % 200 == 0:
            # print(counter.value)
        pbar.n = counter.value
        pbar.refresh()


# modify data processing to calculate coefficient: MFCC, delta MFCC, and delta delta MFCC
def calculate_delta(array):
    # https://github.com/MohamadMerchant/Voice-Authentication-and-Face-Recognition
    rows, cols = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (
            array[index[0][0]]
            - array[index[0][1]]
            + (2 * (array[index[1][0]] - array[index[1][1]]))
        ) / 10
    return deltas

# read in labels
filename2label = {}
for line in open(args.label_path):
    line = line.split()
    filename, label = line[1], line[-1]
    filename2label[filename] = label

def process_audio(filepath_tuple):
    # # read in labels
    # filename2label = {}
    # for line in open(args.label_path):
    #     line = line.split()
    #     filename, label = line[1], line[-1]
    #     filename2label[filename] = label

    index, filepath = filepath_tuple

    # Read audio file
    filename = filepath.split(".")[0]
    if filename not in filename2label:
        # we skip speaker enrollment stage
        return
    label = filename2label[filename]

    sig, rate = sf.read(os.path.join(args.data_path, filepath))

    increment()

    # extract feature, delta of feature, delta delta of feature
    if args.ftype == "silence":
        feat = silence_measure.get_silence(sig, rate)
        feat = feat.reshape((1, 2))
        if args.data_split == "train" or args.data_split == "dev":
            return (feat, label)
        else:
            return (index, feat)

    if args.ftype == "lfcc":
        feat = spafe.features.lfcc.lfcc(
            sig,
            fs=rate,
            num_ceps=20,
            pre_emph=0,
            win_len=0.03,
            win_hop=0.015,
            nfilts=70,
            nfft=1024,
        )
    elif args.ftype == "lfcc-25":
        feat = spafe.features.lfcc.lfcc(
            sig,
            fs=rate,
            num_ceps=20,
            pre_emph=0,
            win_len=0.025,
            win_hop=0.01,
            nfilts=70,
            nfft=1024,
        )
    elif args.ftype == "mfcc":
        feat = spafe.features.mfcc.mfcc(
            sig,
            fs=rate,
            num_ceps=20,
            pre_emph=0,
            win_len=0.03,
            win_hop=0.015,
            nfilts=70,
            nfft=1024,
        )
    elif args.ftype == "bfcc":
        feat = spafe.features.bfcc.bfcc(
            sig,
            fs=rate,
            num_ceps=20,
            pre_emph=0,
            win_len=0.03,
            win_hop=0.015,
            nfilts=70,
            nfft=1024,
        )
    elif args.ftype == "gfcc":
        feat = spafe.features.gfcc.gfcc(
            sig,
            fs=rate,
            num_ceps=20,
            pre_emph=0,
            win_len=0.03,
            win_hop=0.015,
            nfilts=70,
            nfft=1024,
        )
    elif args.ftype == "ngcc":
        feat = spafe.features.ngcc.ngcc(
            sig,
            fs=rate,
            num_ceps=20,
            pre_emph=0,
            win_len=0.03,
            win_hop=0.015,
            nfilts=70,
            nfft=1024,
        )
    else:
        print("Bad feature type!")
    delta_feat = calculate_delta(feat)
    delta_delta_feat = calculate_delta(delta_feat)
    combined_feat = np.hstack((feat, delta_feat, delta_delta_feat))

    if args.data_split == "train" or args.data_split == "dev":
        return (combined_feat, label)
    else:
        return (index, combined_feat)


if __name__ == "__main__":
    counter = multiprocessing.Value(c_int)
    counter_lock = multiprocessing.Lock()

    filepath_list = None

    if args.data_split == "train" or args.data_split == "dev":
        # process data in order of file name (order of file placed in dataset folder)
        filepath_list = os.listdir(args.data_path)
    else:
        # process data in order defined by protocol file
        with open(args.label_path, 'r') as f:
            lines = f.readlines()
        filepath_list = [line.split()[1] + ".flac" for line in lines]

    pbar = tqdm(total=len(filepath_list))
    
    # Multiprocess to prepare data
    a_pool = multiprocessing.Pool(8)
    # feat_label = a_pool.map(process_audio, os.listdir(args.data_path))
    data_list = a_pool.map(process_audio, enumerate(filepath_list))

    pbar.close()

    if args.data_split == "eval":
        # sort data by order
        data_list.sort(key= lambda x: x[0])
        data_list = [data for _, data in data_list]

    # Create folder to save data
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Save output data
    output_path = os.path.join(args.output_path, "{}-{}.pkl".format(args.data_split, args.ftype))
    with open(output_path, "wb") as outfile:
        pickle.dump(data_list, outfile)
        print("dumpped", output_path)
