# from python_speech_features import mfcc
# import sklearn
# import sklearn.preprocessing
from CQCC.cqcc import cqcc
# import scipy.io.wavfile as wav
import soundfile as sf
import os
import numpy as np
import pickle
import argparse
import multiprocessing
import spafe.features.lfcc
import spafe.features.mfcc

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to ASVSpoof data directory. For example, LA/ASVspoof2019_LA_train/flac/')
parser.add_argument("--label_path", required=True, type=str, help='path to label file. For example, LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
parser.add_argument("--output_path", required=True, type=str, help='path to output pickle file. For example, ./data/train.pkl')
# parser.add_argument("--feature_type", required=True, type=str, help='select the feature type. cqcc or mfcc')
args = parser.parse_args()

## modify data processing to calculate LFCC coefficient and MFCC, delta MFCC, and delta delta MFCC

def extract_cqcc(x, fs):
    # INPUT SIGNAL
    x = x.reshape(x.shape[0], 1)  # for one-channel signal 
    # print(x.shape)
    # fs: 16000
    # x: (64244,)
    # PARAMETERS
    B = 96
    fmax = fs/2
    fmin = fmax/2**9
    d = 16
    cf = 19
    ZsdD = 'ZsdD'
    # COMPUTE CQCC FEATURES
    CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec, absCQT = cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD)
    return CQcc, fmax, fmin

def calculate_nfft(samplerate, winlen):
    """Calculates the FFT size as a power of two greater than or equal to
    the number of samples in a single window length.
    
    Having an FFT less than the window length loses precision by dropping
    many of the samples; a longer FFT than the window allows zero-padding
    of the FFT buffer which is neutral in terms of frequency domain conversion.
    :param samplerate: The sample rate of the signal we are working with, in Hz.
    :param winlen: The length of the analysis window in seconds.
    """
    window_length_samples = winlen * samplerate
    nfft = 1
    while nfft < window_length_samples:
        nfft *= 2
    return nfft

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

# read in labels
filename2label = {}
for line in open(args.label_path):
    line = line.split()
    filename, label = line[1], line[-1]
    filename2label[filename] = label

# feats = []

from ctypes import c_int
counter = multiprocessing.Value(c_int)
counter_lock = multiprocessing.Lock()

def increment():
    with counter_lock:
        counter.value += 1
        if counter.value % 200 == 0:
            print(counter.value)

def process_audio(filepath):
    filename = filepath.split('.')[0]
    if filename not in filename2label: # we skip speaker enrollment stage
        return
    label = filename2label[filename]
    # print("filename:", os.path.join(args.data_path, filepath))
    sig, rate = sf.read(os.path.join(args.data_path, filepath))

    # extract lfcc, delta, delta delta
    feat_lfcc = spafe.features.lfcc.lfcc(sig, fs=rate, num_ceps=20, pre_emph=0, win_len=0.03, win_hop=0.015, nfilts=70, nfft=1024)
    # feat_lfcc = sklearn.preprocessing.scale(feat_lfcc)
    delta_feat_lfcc = calculate_delta(feat_lfcc)
    delta_delta_feat_lfcc = calculate_delta(delta_feat_lfcc)
    feat_lfcc_combine = np.hstack((feat_lfcc, delta_feat_lfcc, delta_delta_feat_lfcc))

    # extract mfcc, delta, delta delta
    feat_mfcc = spafe.features.mfcc.mfcc(sig, fs=rate, num_ceps=20, pre_emph=0, win_len=0.03, win_hop=0.015, nfilts=70, nfft=1024)
    delta_feat_mfcc = calculate_delta(feat_mfcc)
    delta_delta_feat_mfcc = calculate_delta(delta_feat_mfcc)
    feat_mfcc_combine = np.hstack((feat_mfcc, delta_feat_mfcc, delta_delta_feat_mfcc))
    # print(feat_mfcc_combine.shape)
    increment()

    return (feat_lfcc_combine, feat_mfcc_combine, label)

chunksize = 2600
total_size = len(os.listdir(args.data_path))

a_pool = multiprocessing.Pool(8)

for n in range(10):

    feats = []

    # a_pool = multiprocessing.Pool(8)

    feats = a_pool.map(process_audio, os.listdir(args.data_path)[n*chunksize : (n+1)*chunksize])

    print("number of instances:", len(feats))

    output_path = args.output_path + "-{}.pkl".format(n)
    with open(output_path, 'wb') as outfile:
        pickle.dump(feats, outfile)
    print("dumpped", output_path)


