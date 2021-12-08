echo "training with train data on features: lfcc"
python3 gmm_train_recur_separate.py --data_path ./data/train/lfcc.pkl \
                    --model_path ./model/lfcc