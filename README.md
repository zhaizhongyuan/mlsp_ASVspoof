# ASVspoof

## install
tested working on python 3.7 environment

```
pip install numpy soundfile spafe sklearn
```

## Feature Extraction
Take MFCC as an example:
assume LA dataset is located at /mnt/LA
```
echo "preparing training data"
python3 data_processing.py --data_path /mnt/LA/ASVspoof2019_LA_train/flac \
                        --output_path ./data/train/ \
                        --label_path /mnt/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
                        --ftype mfcc

echo "preparing dev data"
python3 data_processing.py --data_path /mnt/LA/ASVspoof2019_LA_dev/flac \
                        --output_path ./data/dev/ \
                        --label_path /mnt/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
                        --ftype mfcc

echo "preparing test data"
python3 data_processing.py --data_path /mnt/LA/ASVspoof2019_LA_eval/flac \
                        --output_path ./data/test/ \
                        --label_path /mnt/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
                        --ftype mfcc
```

The saved pickle file has the format: [(feature[framecountx60], label[bonafide/spoof]) x N instances]

**PLEASE save data pickles under directories named with feature type.**

**After preprocessing to pickles, please scp pickles to /mnt/LA/preprocessed/...**

## GMM TRAINING
Take MFCC as an example:

```
echo "training on features: mfcc"
python3 gmm_train.py --data_path ./data/train/mfcc.pkl \
                    --model_path ./model/mfcc
```
**PLEASE save model pickles under directories named with feature type.**
**After training to pickles, please scp pickles to /mnt/LA/models/...**

## GMM TESTING
Take MFCC as an example:

```
echo "testing on features: mfcc"
python3 gmm_test.py --data_path ./data/dev/mfcc.pkl \
                    --model_path_bon ./model/mfcc/bon.gmm \
                    --model_path_sp ./model/mfcc/sp.gmm

echo "evaluating on features: mfcc"
python3 gmm_test.py --data_path ./data/test/mfcc.pkl \
                    --model_path_bon ./model/mfcc/bon.gmm \
                    --model_path_sp ./model/mfcc/sp.gmm
```