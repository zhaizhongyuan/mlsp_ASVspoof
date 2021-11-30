# ASVspoof

## install
tested working on python 3.7 environment

```
pip install numpy soundfile spafe sklearn
```


## Feature Extraction
assume LA dataset is located at ../LA

```
python3 data_processing.py --data_path ../LA/ASVspoof2019_LA_train/flac --output_path ./data/train --label_path ../LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
```
```
python3 data_processing.py --data_path ../LA/ASVspoof2019_LA_dev/flac --output_path ./data/dev --label_path ../LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt
```
```
python3 data_processing.py --data_path ../LA/ASVspoof2019_LA_eval/flac --output_path ./data/eval --label_path ../LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
```

The saved pickle file has the format: [(lfcc_vec[framecountx60], mfcc_vec[framecountx60], label[bonafide/spoof]) x N instances]

## GMM TRAINING

### MFCC
```
python3 gmm_train.py --data_path ./data/train --model_path ./model/mfcc_gmm_ --feature_type mfcc
```

### LFCC
```
python3 gmm_train.py --data_path ./data/train --model_path ./model/lfcc_gmm_ --feature_type lfcc
```

## GMM TESTING

### MFCC
```
python gmm_test.py --data_path ./data/dev --model_path_bon ./model/mfcc_gmm_bon_epoch9.gmm --model_path_sp ./model/mfcc_gmm_sp_epoch9.gmm --feature_type mfcc
```

```
python gmm_test.py --data_path ./data/test --model_path_bon ./model/mfcc_gmm_bon_epoch9.gmm --model_path_sp ./model/mfcc_gmm_sp_epoch9.gmm --feature_type mfcc
```

### LFCC
```
python gmm_test.py --data_path ./data/dev --model_path_bon ./model/lfcc_gmm_bon_epoch9.gmm --model_path_sp ./model/lfcc_gmm_sp_epoch9.gmm --feature_type lfcc
```

```
python gmm_test.py --data_path ./data/test --model_path_bon ./model/lfcc_gmm_bon_epoch9.gmm --model_path_sp ./model/lfcc_gmm_sp_epoch9.gmm --feature_type lfcc
```
