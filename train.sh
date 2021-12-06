echo "training with train data on features: mfcc"
python3 gmm_train.py --data_path ./data/train/mfcc.pkl \
                    --model_path ./model/mfcc