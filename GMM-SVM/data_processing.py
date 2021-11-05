from python_speech_features import mfcc
from CQCC.cqcc import cqcc
import scipy.io.wavfile as wav
import soundfile as sf
import os
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to ASVSpoof data directory. For example, LA/ASVspoof2019_LA_train/flac/')
parser.add_argument("--label_path", required=True, type=str, help='path to label file. For example, LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt')
parser.add_argument("--output_path", required=True, type=str, help='path to output pickle file. For example, ./data/train.pkl')
# parser.add_argument("--feature_type", required=True, type=str, help='select the feature type. cqcc or mfcc')
args = parser.parse_args()

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

# read in labels
filename2label = {}
for line in open(args.label_path):
    line = line.split()
    filename, label = line[1], line[-1]
    filename2label[filename] = label

feats = []
for filepath in os.listdir(args.data_path):
    filename = filepath.split('.')[0]
    if filename not in filename2label: # we skip speaker enrollment stage
        continue
    label = filename2label[filename]
    print("filename:", os.path.join(args.data_path, filepath))
    sig, rate = sf.read(os.path.join(args.data_path, filepath))
    print(sig.shape)
    print("rate:", rate)
    feat_cqcc, fmax, fmin = extract_cqcc(sig, rate)
    print("feat cqcc:", feat_cqcc.shape)
    numframes = feat_cqcc.shape[0]
    winstep = 0.005
    winlen =  (len(sig) - winstep*rate*(numframes-1))/rate
    nfft = calculate_nfft(rate, winlen)
    feat_mfcc = mfcc(sig,rate,winlen=winlen,winstep=winstep, lowfreq=fmin,highfreq=fmax, nfft=nfft)      # number of frames * number of cep
    # if args.feature_type == "cqcc":
    #     feat = extract_cqcc(sig, rate)
    # elif args.feature_type == "mfcc":
    #     feat = mfcc(sig, rate)
    print("feat mfcc:", feat_mfcc.shape)
    feats.append((feat_cqcc, feat_mfcc, label))
    if len(feats) % 1000 == 0:
        print(len(feats))

print("number of instances:", len(feats))

with open(args.output_path, 'wb') as outfile:
    pickle.dump(feats, outfile)


