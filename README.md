# ASVspoof

## install
tested working on python 3.7 environment

```
pip install numpy soundfile spafe sklearn
```

## directory structure
The code files are divided into three categories: Code for Single Feature GMM Base Model, Code for Frame-level (ensemble) Fusion GMM model, Code for Instance-level / File-level Fusion GMM model.

The `tDCF_python/` is a official tool published by ASVspoof 2021 to calculate Equal Error Rate (EER) and min t-DCF score.

For all the code file below, please checkout specific argument format with `-h` flag.

### GMM
`gmm_data_processing.py` will extract cepstral coefficients from audio file data and store them in Python pickle file under folder `data/`

`gmm_train_recur.py` will train GMM base model chunk-by-chunk recursively, since training set will not fir into memory for most computers, the trained GMM model will be pickled and stored in `model/`

`gmm_test.py` will test the trained GMM model on evaluation dataset and stored an specific format text in `output/` for t-DCF and EER metrics.

`preprocess.sh`, `train.sh`, and `test.sh` are Shell script to help you run data processing, training, and testing on base GMM model.

### Frame-level / Ensemble
`gmm_ensemble_preprocessing.py` will take an specific GMM base model and a dataset, generate the base model's score output, and then store them into pickle files for frame-level fusion use in `ensemble/`

`gmm_ensemble_train.py` and `gmm_ensemble_train_recur.py` will train the frame-level fusion GMM model based on data provided. The normal train will handle smaller size GMM models, while the recur train file is needed when training large number component GMM models.

`gmm_ensemble_test.py` will test the trained frame-level GMM model on evaluation dataset and stored an specific format text in `output/` for t-DCF and EER metrics.

### Instance-level / Filelevel
`gmm_filelevel_preprocessing.py` will take an specific frame-level fusion GMM model and a dataset, generate the frame-level fusion model's score output, and then store them into pickle files for instance-level fusion use in `ensemble/`. Later, manually add or remove the frame-level model's output score you would like to fuse into a new instance-level model. Generally, the data for instance-level is stored in `experiment/`.

`gmm_filelevel_train.py` will train the instance-level fusion GMM model based on data provided and store the model in `model/`.

`gmm_filelevel_test.py` will test the trained instance-level GMM model on evaluation dataset and stored an specific format text in `output/` for t-DCF and EER metrics.

`silence_measure.py` will calculate the leading and trailing silence frame count, in order to generate the 'silence' feature. This functions can be invoked when passing `--ftype silence` into data preprocessing Python file.

### Tools
`score_eval.py` is a wrapper program over tDCF_python to help calculate evaluation metrics, specific score text generated by various `*_test.py` can be read from `output/`

# Sample Commands to Run the Code
Assuming ASVspoof 2019's Logical Access dataset is located at `./LA`

## GMM Feature Extraction
Take MFCC as an example:
assume LA dataset is located at /mnt/LA
```
echo "extracting features: mfcc"
echo "preparing training data"
python3 gmm_data_processing.py --data_split train \
                        --data_path ./LA/ASVspoof2019_LA_train/flac \
                        --output_path ./data/train/ \
                        --label_path ./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
                        --ftype mfcc

echo "preparing dev data"
python3 gmm_data_processing.py --data_split dev \
                        --data_path ./LA/ASVspoof2019_LA_dev/flac \
                        --output_path ./data/dev/ \
                        --label_path ./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
                        --ftype mfcc

echo "preparing test data"
python3 gmm_data_processing.py --data_split eval \
                        --data_path ./LA/ASVspoof2019_LA_eval/flac \
                        --output_path ./data/test/ \
                        --label_path ./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
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

python gmm_test.py --data_path ./data/eval/eval-mfcc.pkl \
                         --model_path_bon ./model/mfcc/bon.gmm \
                         --model_path_sp ./model/mfcc/sp.gmm \
                         --label_path ./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
                         --output_path ./output/mfcc-eval.text

python score_eval.py --cm_score ./output/mfcc-eval.text \
                    --asv_score ./tDCF_python/scores/ASVspoof2019_LA_eval_asv_scores.txt
```

## Frame-level Data Preprocessing
Take LFCC as an example:
```
python gmm_ensemble_preprocess.py --data_split eval --data_path ./data/eval/eval-lfcc.pkl --model_path ./model/lfcc --model_name lfcc --output_path ./ensemble/lfcc
```

## Frame-level Training
Take LFCC as an example:
```
python gmm_ensemble_train_recur.py --data_dir ./experiment/lfcc/train --model_path ./model/ensemble-lfcc
```


## Frame-level Testing
Take LFCC as an example:
```
python gmm_filelevel_preprocess.py --data_split train --data_dir ./experiment/lfcc/train --model_path ./model/ensemble-lfcc --output_path ./ensemble/ensemble-lfcc --model_name ensemble-lfcc
```

## Instance-level Data Preprocessing
Take LFCC as an example:
```
python gmm_ensemble_preprocess.py --data_split eval --data_path ./data/eval/eval-lfcc.pkl --model_path ./model/lfcc --model_name lfcc --output_path ./ensemble/lfcc
```

## Instance-level Training
Take LFCC as an example:
```
python gmm_filelevel_train.py --data_dir ./experiment/filelevel-lfcc/train --model_path ./model/filelevel-lfcc
```


## Instance-level Testing
Take LFCC as an example:
```
python gmm_filelevel_test.py --data_dir ./experiment/filelevel-lfcc/eval --model_path_bon ./model/filelevel-lfcc/bon.gmm --model_path_sp ./model/filelevel-lfcc/sp.gmm --label_path ./LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt --output_path ./output/filelevel-lfcc.txt
```
