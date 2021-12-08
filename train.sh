echo "training with train data on features: mfcc"
# python3 gmm_train.py --data_path ./data/train/mfcc.pkl \
#                     --model_path ./model/mfcc

python3 gmm_train_recur_separate.py --data_path ./data/train/train-mfcc.pkl \
                                    --model_path ./model/mfcc